"""Корпус KPI-утверждений из проспектов SaaS-компаний (SEC EDGAR, формы S-1 / 424B4).

Что делает скрипт:
1. Ищет проспекты 2019–2025 через полнотекстовый поиск EDGAR (efts.sec.gov) по фразам
   «annual recurring revenue», «dollar-based net retention rate» и т. п.
2. Оставляет компании с SIC 7370–7379 (software / computer services), по одной заявке
   на компанию: финальный проспект 424B4, иначе последняя S-1/A, иначе S-1.
3. Скачивает HTML (кэш в data/external/sec/raw/, не коммитится), вытаскивает текст,
   режет на предложения и прозрачным rule-based экстрактором находит предложения про
   SaaS-KPI: ARR, MRR, NRR (DBNRR), GRR, число клиентов, крупные клиенты, churn, рост выручки.
4. Сохраняет компактный производный корпус в data/reference/sec_kpi_claims.csv (< 500 KB)
   и выборку для ручной проверки экстрактора в data/external/sec/validation_sample.csv.

Тексты SEC EDGAR — public domain. Правила fair access: User-Agent из переменной окружения
SEC_USER_AGENT (по умолчанию общий идентификатор проекта), не больше 5 запросов в секунду.
Общего User-Agent хватает только для поиска efts.sec.gov; архив www.sec.gov отвечает 403,
пока в SEC_USER_AGENT нет контакта (имя и e-mail исследователя, только через окружение).

Запуск из корня репозитория:
    uv run python scripts/fetch_sec_claims.py                 # всё по умолчанию
    uv run python scripts/fetch_sec_claims.py --max-companies 40 --offline
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "external" / "sec"
RAW_DIR = CACHE_DIR / "raw"
SEARCH_DIR = CACHE_DIR / "search"
OUT_CSV = ROOT / "data" / "reference" / "sec_kpi_claims.csv"
SAMPLE_CSV = CACHE_DIR / "validation_sample.csv"
FILINGS_CSV = CACHE_DIR / "filings_selected.csv"

SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{adsh}/{filename}"
DEFAULT_UA = "dd-verifier-student-project research"
MIN_INTERVAL_S = 0.25  # ≤ 4 запросов в секунду, с запасом к лимиту SEC в 10 req/s
UA_HELP = (
    "SEC вернул 403 «Undeclared Automated Tool»: архив www.sec.gov требует User-Agent "
    "с контактом, например:\n"
    '  SEC_USER_AGENT="Имя Фамилия contact@example.org" '
    "uv run python scripts/fetch_sec_claims.py\n"
    "Полнотекстовый поиск (efts.sec.gov) работает и с общим User-Agent, документы — нет. "
    "Контакт задаётся только через переменную окружения и в репозиторий не попадает."
)

QUERIES = [
    '"annual recurring revenue"',
    '"dollar-based net retention rate"',
    '"net revenue retention"',
    '"net dollar retention"',
    '"dollar-based net expansion rate"',
    '"gross revenue retention"',
]
FORMS = "S-1,424B4"
START_DATE, END_DATE = "2019-01-01", "2025-12-31"
ALLOWED_FILE_TYPES = {"S-1", "S-1/A", "424B4"}
FORM_PRIORITY = {"424B4": 0, "S-1/A": 1, "S-1": 2}

TEXT_MAX_CHARS = 320  # обычные предложения
DEF_TEXT_MAX_CHARS = 900  # определения храним длиннее: их сравниваем между компаниями

# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------


class SecClient:
    """Вежливый клиент EDGAR: свой User-Agent, пауза между запросами, ретраи на 429/5xx."""

    def __init__(self, user_agent: str, offline: bool = False) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip"})
        self.offline = offline
        self._last = 0.0
        self.n_requests = 0

    def get(self, url: str) -> requests.Response:
        if self.offline:
            raise RuntimeError(f"offline mode: {url} is not cached")
        for attempt in range(5):
            wait = MIN_INTERVAL_S - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.n_requests += 1
            resp = self.session.get(url, timeout=60)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 403 and b"Undeclared Automated Tool" in resp.content:
                raise SystemExit(UA_HELP)
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** (attempt + 1))
                continue
            resp.raise_for_status()
        raise RuntimeError(f"failed after retries: {url}")


def search_hits(client: SecClient, query: str, max_pages: int) -> list[dict]:
    """Все страницы выдачи полнотекстового поиска (кэшируются в data/external/sec/search/)."""
    hits: list[dict] = []
    for page in range(max_pages):
        key = hashlib.sha1(f"{query}|{page}".encode()).hexdigest()[:16]
        cache = SEARCH_DIR / f"{key}.json"
        if cache.exists():
            data = json.loads(cache.read_text())
        else:
            url = (
                f"{SEARCH_URL}?q={quote(query)}&forms={FORMS}&dateRange=custom"
                f"&startdt={START_DATE}&enddt={END_DATE}&from={page * 100}"
            )
            data = client.get(url).json()
            cache.write_text(json.dumps(data))
        page_hits = data.get("hits", {}).get("hits", [])
        for h in page_hits:
            h["_query"] = query
        hits.extend(page_hits)
        total = data.get("hits", {}).get("total", {}).get("value", 0)
        if not page_hits or (page + 1) * 100 >= total:
            break
    return hits


def select_filings(hits: list[dict], max_companies: int) -> pd.DataFrame:
    """Одна заявка на компанию; компании ранжируются по числу разных KPI-запросов."""
    rows = []
    for h in hits:
        src = h["_source"]
        sics = src.get("sics") or []
        if src.get("file_type") not in ALLOWED_FILE_TYPES:
            continue  # exhibits (EX-10.x и т. п.) пропускаем
        if not sics or not sics[0].startswith("737"):
            continue  # только software / computer services
        adsh, filename = h["_id"].split(":", 1)
        name = src["display_names"][0]
        rows.append(
            {
                "cik": int(src["ciks"][0]),
                "company": re.sub(r"\s+\(CIK \d+\)$", "", name).strip(),
                "sic": sics[0],
                "form": src["file_type"],
                "filing_date": src["file_date"],
                "adsh": adsh,
                "filename": filename,
                "query": h["_query"],
            }
        )
    df = pd.DataFrame(rows).drop_duplicates()
    n_queries = df.groupby("cik")["query"].nunique().rename("n_queries")
    df = df.drop(columns="query").drop_duplicates().join(n_queries, on="cik")
    df["form_rank"] = df["form"].map(FORM_PRIORITY)
    df = df.sort_values(["cik", "form_rank", "filing_date"], ascending=[True, True, False])
    best = df.groupby("cik").head(1)
    best = best.sort_values(["n_queries", "filing_date"], ascending=[False, False])
    best = best.head(max_companies).reset_index(drop=True)
    best["url"] = [
        ARCHIVE_URL.format(cik=r.cik, adsh=r.adsh.replace("-", ""), filename=r.filename)
        for r in best.itertuples()
    ]
    return best.drop(columns="form_rank")


def fetch_document(client: SecClient, row: pd.Series) -> str:
    path = RAW_DIR / f"{row['adsh']}_{row['filename']}"
    if not path.exists():
        path.write_bytes(client.get(row["url"]).content)
    return path.read_bytes().decode("utf-8", errors="replace")


# --------------------------------------------------------------------------------------
# HTML → текст → предложения
# --------------------------------------------------------------------------------------

BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "hr"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"script", "style"}:
            self._skip += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")
        elif tag == "td":
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self._skip = max(0, self._skip - 1)
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_blocks(raw: str) -> list[str]:
    """Текстовые блоки (абзацы) документа с нормализованными пробелами."""
    parser = _TextExtractor()
    parser.feed(raw)
    text = html.unescape("".join(parser.parts))
    text = text.replace("\xa0", " ").replace("​", "")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("—", " - ").replace("–", "-")
    blocks = []
    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) >= 40:
            blocks.append(line)
    return blocks


_ABBREV = re.compile(
    r"\b(Inc|Corp|Co|Ltd|No|U\.S|e\.g|i\.e|vs|approx|Mr|Ms|Dr|St|Jan|Feb|Mar|Apr|Jun|Jul|Aug"
    r"|Sep|Sept|Oct|Nov|Dec|L\.P|N\.A|S\.A)\.",
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"(]?[A-Z•])")


def split_sentences(block: str) -> list[str]:
    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", "<DOT>"), block)
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)
    out = []
    for s in _SENT_SPLIT.split(protected):
        s = s.replace("<DOT>", ".").strip(" •·-")
        if len(s) >= 40:
            out.append(s)
    return out


# --------------------------------------------------------------------------------------
# Rule-based экстрактор
# --------------------------------------------------------------------------------------

# Семейства метрик в порядке приоритета: если в предложении несколько, основным
# считается первое из этого списка (NRR важнее упоминания ARR в его определении и т. д.).
FAMILY_PATTERNS: dict[str, re.Pattern] = {
    "nrr": re.compile(
        r"net (?:dollar |revenue )?(?:retention|expansion)|dollar[- ]based net (?:revenue )?"
        r"(?:retention|expansion)|\bNRR\b|\bNDR\b|\bDBNR+\b|\bDBNE?R\b",
        re.I,
    ),
    "grr": re.compile(
        r"gross (?:dollar |revenue )?retention|dollar[- ]based gross retention|\bGRR\b", re.I
    ),
    "churn": re.compile(r"\bchurn|\battrition rate|\blogo retention|customer retention rate", re.I),
    "large_customers": re.compile(
        r"customers? (?:that|who|with|whose|which|contributing|generating|representing|paying)"
        r"[^;]{0,80}?(?:\$\s?\d[\d,.]*\s?(?:million|thousand|k\b)?|\$\d)",
        re.I,
    ),
    "arr": re.compile(r"annual(?:ized)? recurring revenue|\bARR\b", re.I),
    "mrr": re.compile(r"monthly recurring revenue|\bMRR\b", re.I),
    "customers": re.compile(
        r"\d[\d,]*\+?\s+(?:[A-Za-z-]+\s+){0,3}?(?:customers|clients|subscribers)\b"
        r"|(?:customers|clients) (?:grew|increased|rose) (?:from [\d,]+ )?to (?:over |more than "
        r"|approximately )?\d",
        re.I,
    ),
    "revenue_growth": re.compile(
        r"revenue[^;]{0,60}?(?:grew|growth|increased|increase)[^;]{0,80}?\d+(?:\.\d+)?\s?%"
        r"|revenue[^;]{0,60}?(?:grew|increased)[^;]{0,40}?\d+(?:\.\d+)?\s?(?:x|times)\b",
        re.I,
    ),
}
FAMILY_UNIT = {
    "nrr": "percent",
    "grr": "percent",
    "churn": "percent",
    "large_customers": "count",
    "arr": "usd",
    "mrr": "usd",
    "customers": "count",
    "revenue_growth": "percent",
}

DEFINITION_RE = re.compile(
    r"\bwe (?:calculate|define|compute|measure|determine|derive)\b"
    r"|\b(?:is|are|was|were) (?:calculated|defined|computed|measured|determined|derived)\b"
    r"|\bdefined as\b|\bwe refer to\b|\bwe use the term\b|\bwe consider\b[^.]{0,60}\bcustomer"
    r"|\bwe (?:first )?(?:identify|divide)\b|\bwe then\b|\bdivided by\b",
    re.I,
)

QUALIFIERS = [
    ("more_than", r"more than|over|greater than|in excess of|exceeding|exceeded|above|upwards of"),
    ("at_least", r"at least|no less than"),
    ("about", r"approximately|about|around|roughly|nearly|almost|close to|~"),
    ("less_than", r"less than|under|below|fewer than"),
    ("up_to", r"up to"),
]
_QUAL_RE = [(name, re.compile(rf"(?:{pat})\s*$", re.I)) for name, pat in QUALIFIERS]

NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
MONEY_RE = re.compile(rf"\$\s?({NUM})\s*(billion|million|thousand|[BMK]\b)?", re.I)
PCT_RE = re.compile(rf"({NUM})\s?(%|percent\b)", re.I)
MULT_RE = re.compile(rf"\b({NUM})\s?(x|times)\b", re.I)
COUNT_RE = re.compile(rf"(?<![$\d.,])({NUM})(\+)?(?![\d.,]*\s?(?:%|percent|million|billion|x\b))")
SCALE = {"billion": 1e9, "b": 1e9, "million": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}
YEAR_RE = re.compile(r"^(19|20)\d\d$")

PERIOD_PATTERNS = [
    (
        "as_of",
        r"as of (?:the end of )?(?:(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December) \d{1,2}, \d{4}|fiscal (?:year )?\d{4}|"
        r"(?:January|February|March|April|May|June|July|August|September|October|November|"
        r"December) \d{4}|the end of (?:each|the) [a-z ]+)",
    ),
    (
        "fiscal_period",
        r"(?:for|during|in) (?:the |each of the )?(?:fiscal )?(?:years?|three months|six months|"
        r"nine months|quarters?|twelve months|12 months)(?: ended| ending)[^.;]{0,40}?\d{4}"
        r"|(?:in|for|during) (?:fiscal|calendar) (?:year )?\d{4}|\bfiscal (?:year )?\d{4}\b"
        r"|\bFY ?\d{2,4}\b",
    ),
    (
        "trailing",
        r"trailing (?:twelve|12|four|4)[- ]months?|\bTTM\b|(?:prior|preceding|previous|"
        r"trailing) (?:twelve|12)[- ]month|twelve months prior|12 months prior|"
        r"same (?:date|period) (?:in|of) the prior year",
    ),
    ("yoy", r"year[- ]over[- ]year|compared to the (?:prior|previous|same) (?:year|period)"),
]
_PERIOD_RE = [(name, re.compile(pat, re.I)) for name, pat in PERIOD_PATTERNS]


@dataclass
class Number:
    value: float
    unit: str
    start: int
    end: int
    raw: str
    plus: bool = False


@dataclass
class Claim:
    metric_family: str
    other_families: list[str] = field(default_factory=list)
    value: float | None = None
    values: list[float] = field(default_factory=list)
    unit: str | None = None
    qualifier: str | None = None
    threshold_usd: float | None = None
    period_phrase: str | None = None
    period_type: str | None = None
    is_definition: bool = False


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def find_numbers(sentence: str) -> list[Number]:
    nums: list[Number] = []
    taken: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= s or a >= e for s, e in taken)

    for m in MONEY_RE.finditer(sentence):
        scale = SCALE.get((m.group(2) or "").lower(), 1.0)
        nums.append(Number(_to_float(m.group(1)) * scale, "usd", m.start(), m.end(), m.group(0)))
        taken.append((m.start(), m.end()))
    for m in PCT_RE.finditer(sentence):
        if free(m.start(), m.end()):
            nums.append(Number(_to_float(m.group(1)), "percent", m.start(), m.end(), m.group(0)))
            taken.append((m.start(), m.end()))
    for m in MULT_RE.finditer(sentence):
        if free(m.start(), m.end()):
            nums.append(Number(_to_float(m.group(1)), "multiple", m.start(), m.end(), m.group(0)))
            taken.append((m.start(), m.end()))
    for m in COUNT_RE.finditer(sentence):
        if not free(m.start(), m.end()) or YEAR_RE.match(m.group(1)):
            continue
        tail = sentence[m.end() : m.end() + 12].lower()
        if re.match(r"\s?(?:days?|months?|years?|quarters?)\b", tail):
            continue  # «12 months», «30 days» — не значение метрики
        nums.append(
            Number(_to_float(m.group(1)), "count", m.start(), m.end(), m.group(0), bool(m.group(2)))
        )
    return sorted(nums, key=lambda n: n.start)


def qualifier_before(sentence: str, num: Number) -> str:
    if num.plus or sentence[num.end : num.end + 1] == "+":
        return "more_than"
    left = sentence[max(0, num.start - 30) : num.start]
    for name, rx in _QUAL_RE:
        if rx.search(left):
            return name
    return "exact"


def detect_period(sentence: str) -> tuple[str | None, str | None]:
    found = [(name, rx.search(sentence)) for name, rx in _PERIOD_RE]
    found = [(name, m) for name, m in found if m]
    if not found:
        return None, None
    types = "+".join(name for name, _ in found)
    return found[0][1].group(0), types


def extract_claim(sentence: str) -> Claim | None:
    """Правило: семейство KPI + (число нужной единицы или определение метрики)."""
    matches = {fam: rx.search(sentence) for fam, rx in FAMILY_PATTERNS.items()}
    families = [fam for fam, m in matches.items() if m]
    if not families:
        return None
    family = families[0]
    is_def = bool(DEFINITION_RE.search(sentence))
    nums = find_numbers(sentence)
    unit = FAMILY_UNIT[family]
    anchor = matches[family].start()

    threshold = None
    if family == "large_customers":
        money = [n for n in nums if n.unit == "usd"]
        threshold = money[0].value if money else None
    candidates = [n for n in nums if n.unit == unit]
    if family == "revenue_growth":
        candidates += [n for n in nums if n.unit == "multiple"]
    if family == "customers":
        candidates = [n for n in candidates if n.value >= 10]  # «3 customers» — шум
    if not candidates and not is_def:
        return None
    claim = Claim(metric_family=family, other_families=families[1:], is_definition=is_def)
    if candidates:
        # значение — ближайшее к ключевому слову (сначала справа, затем слева)
        primary = min(candidates, key=lambda n: (n.start < anchor, abs(n.start - anchor)))
        if family in {"customers", "large_customers"}:
            # «we had 1,200 customers»: число обычно стоит перед словом customers
            primary = min(candidates, key=lambda n: abs(n.end - anchor))
        claim.value = primary.value
        claim.values = [n.value for n in candidates]
        claim.unit = primary.unit
        claim.qualifier = qualifier_before(sentence, primary)
    claim.threshold_usd = threshold
    claim.period_phrase, claim.period_type = detect_period(sentence)
    return claim


# --------------------------------------------------------------------------------------
# Сборка корпуса
# --------------------------------------------------------------------------------------


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9%$]+", " ", s.lower()).strip()


def build_rows(filing: pd.Series, blocks: list[str]) -> tuple[list[dict], list[dict]]:
    """Возвращает (извлечённые строки, отклонённые предложения-кандидаты для оценки recall)."""
    seen: dict[str, dict] = {}
    rejected: list[dict] = []
    broad = re.compile(r"recurring revenue|retention|churn|customers", re.I)
    for b_idx, block in enumerate(blocks):
        sentences = split_sentences(block)
        for s_idx, sent in enumerate(sentences):
            if len(sent) > 1500 or sum(c.isalpha() for c in sent) < 0.5 * len(sent):
                continue
            claim = extract_claim(sent)
            key = _norm(sent)
            if claim is None:
                if broad.search(sent) and re.search(r"\d", sent):
                    rejected.append({"company": filing["company"], "text": sent[:TEXT_MAX_CHARS]})
                continue
            if key in seen:
                seen[key]["n_occurrences"] += 1
                continue
            text = sent
            if claim.is_definition:
                # определение часто продолжается в 1–2 следующих предложениях
                text = " ".join([sent, *sentences[s_idx + 1 : s_idx + 3]])
                limit = DEF_TEXT_MAX_CHARS
            else:
                limit = TEXT_MAX_CHARS
            seen[key] = {
                "company": filing["company"],
                "cik": filing["cik"],
                "form": filing["form"],
                "filing_date": filing["filing_date"],
                "url": filing["url"],
                "block_idx": b_idx,
                "metric_family": claim.metric_family,
                "other_families": ";".join(claim.other_families),
                "value": claim.value,
                "values": ";".join(f"{v:g}" for v in claim.values[:6]),
                "unit": claim.unit,
                "qualifier": claim.qualifier,
                "threshold_usd": claim.threshold_usd,
                "period_phrase": claim.period_phrase,
                "period_type": claim.period_type,
                "is_definition": claim.is_definition,
                "n_occurrences": 1,
                "text_truncated": len(text) > limit,
                "text": text[:limit],
            }
    return list(seen.values()), rejected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-companies", type=int, default=50)
    ap.add_argument("--max-pages", type=int, default=6, help="страниц по 100 хитов на запрос")
    ap.add_argument("--offline", action="store_true", help="только кэш, без сети")
    ap.add_argument("--sample-size", type=int, default=60, help="размер выборки для разметки")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for d in (RAW_DIR, SEARCH_DIR, OUT_CSV.parent):
        d.mkdir(parents=True, exist_ok=True)
    client = SecClient(os.environ.get("SEC_USER_AGENT", DEFAULT_UA), offline=args.offline)

    hits: list[dict] = []
    for q in QUERIES:
        q_hits = search_hits(client, q, args.max_pages)
        print(f"search {q}: {len(q_hits)} hits", file=sys.stderr)
        hits.extend(q_hits)
    filings = select_filings(hits, args.max_companies)
    filings.to_csv(FILINGS_CSV, index=False)
    print(f"selected {len(filings)} filings", file=sys.stderr)

    all_rows: list[dict] = []
    all_rejected: list[dict] = []
    for i, filing in filings.iterrows():
        raw = fetch_document(client, filing)
        rows, rejected = build_rows(filing, html_to_blocks(raw))
        all_rows.extend(rows)
        all_rejected.extend(rejected)
        print(f"[{i + 1}/{len(filings)}] {filing['company']}: {len(rows)} rows", file=sys.stderr)

    df = pd.DataFrame(all_rows)
    df.insert(0, "claim_id", [f"sec{i:05d}" for i in range(len(df))])
    df.to_csv(OUT_CSV, index=False)
    size_kb = OUT_CSV.stat().st_size / 1024
    print(f"wrote {OUT_CSV.relative_to(ROOT)}: {len(df)} rows, {size_kb:.0f} KB", file=sys.stderr)

    # Выборка для ручной проверки: извлечённые (precision) + отклонённые кандидаты (пропуски).
    rng = random.Random(args.seed)
    take = rng.sample(range(len(df)), min(args.sample_size, len(df)))
    sample = df.iloc[sorted(take)][["claim_id", "company", "metric_family", "value", "unit"]]
    sample = sample.assign(
        qualifier=df.loc[sample.index, "qualifier"],
        is_definition=df.loc[sample.index, "is_definition"],
        text=df.loc[sample.index, "text"],
        sample_type="extracted",
    )
    rej = pd.DataFrame(rng.sample(all_rejected, min(30, len(all_rejected))))
    rej = rej.assign(claim_id=[f"rej{i:03d}" for i in range(len(rej))], sample_type="rejected")
    pd.concat([sample, rej], ignore_index=True).to_csv(SAMPLE_CSV, index=False)
    print(f"client requests: {client.n_requests}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
