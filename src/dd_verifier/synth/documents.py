"""Рендеринг документов компании на английском: дек (слайды), пресс-релизы, меморандум CFO.

Каждое утверждение — отдельный буллет или предложение; цитата в эталоне — точная подстрока текста.
"""

from __future__ import annotations

import numpy as np

from dd_verifier.synth.claims import ClaimSpec

TEMPLATES: dict[tuple[str, str], dict[str, list[str]]] = {
    ("arr", "num"): {
        "deck": ["ARR of {q}{v} {p}", "{q}{v} ARR {p}", "Annual recurring revenue: {q}{v} {p}"],
        "press": [
            "{company} reached {q}{v} in annual recurring revenue {p}.",
            "The company's ARR stood at {q}{v} {p}.",
        ],
    },
    ("mrr", "num"): {
        "deck": ["MRR of {q}{v} {p}", "Monthly recurring revenue: {q}{v} {p}"],
        "press": ["Monthly recurring revenue reached {q}{v} {p}."],
    },
    ("revenue", "num"): {
        "deck": ["Revenue of {q}{v} {p}", "{q}{v} in revenue {p}"],
        "press": ["{company} generated {q}{v} in revenue {p}."],
    },
    ("acv", "num"): {
        "deck": ["Average contract value of {q}{v} {p}", "ACV: {q}{v} {p}"],
        "press": ["The average customer contract was worth {q}{v} per year {p}."],
    },
    ("customers", "num"): {
        "deck": [
            "{q}{v} {noun} use {company} every day",
            "{q}{v} paying {noun} {p}",
            "Trusted by {q}{v} {noun}",
        ],
        "press": [
            "{company} now serves {q}{v} {noun}.",
            "The platform is used by {q}{v} {noun}.",
            "{company} served {q}{v} paying {noun} {p}.",
        ],
    },
    ("customers_growth", "mult"): {
        "deck": ["Customer count up {v} {p}"],
        "press": ["The number of paying {noun} grew {v} {p}."],
    },
    ("customers_growth", "pct"): {
        "deck": ["Customer base grew {v} {p}", "+{v} paying {noun} {p}"],
        "press": ["{company} grew its customer base by {v} {p}."],
    },
    ("customers_growth", "word"): {
        "deck": ["Customer base {v} {p}"],
        "press": ["{company} {v} its customer base {p}."],
    },
    ("arr_growth", "mult"): {
        "deck": ["ARR grew {v} {p}", "{v} ARR growth {p}"],
        "press": ["ARR increased {v} {p}.", "{company} grew ARR {v} {p}."],
    },
    ("arr_growth", "pct"): {
        "deck": ["ARR grew {v} {p}", "+{v} ARR growth {p}"],
        "press": ["{company} grew its ARR by {v} {p}.", "Annual recurring revenue rose {v} {p}."],
    },
    ("arr_growth", "word"): {
        "deck": ["ARR {v} {p}"],
        "press": ["{company} {v} its annual recurring revenue {p}."],
    },
    ("nrr", "num"): {
        "deck": ["Net revenue retention of {q}{v} {p}", "NRR: {q}{v} {p}"],
        "press": ["Net revenue retention reached {q}{v} {p}."],
    },
    ("grr", "num"): {
        "deck": ["Gross revenue retention of {q}{v} {p}", "GRR: {q}{v} {p}"],
        "press": ["Gross revenue retention stood at {q}{v} {p}."],
    },
    ("churn", "num"): {
        "deck": ["Monthly logo churn of {q}{v} {p}", "Average monthly churn: {q}{v} {p}"],
        "press": ["Monthly logo churn averaged {q}{v} {p}."],
    },
    ("top10", "num"): {
        "deck": [
            "Top-10 customers account for {q}{v} of revenue",
            "Top-10 customer concentration: {q}{v} of trailing 12-month revenue",
        ],
        "press": ["The ten largest customers account for {q}{v} of revenue."],
    },
    ("forecast", "num"): {
        "deck": ["On track to reach {v} ARR by the end of 2026"],
        "press": ["{company} expects to reach {v} in ARR by the end of 2026."],
    },
    ("opinion", "num"): {
        "deck": ["{company} is the leading {category} platform for {noun}"],
        "press": [
            '"We are building the best {category} platform for {noun}," said the company\'s CEO.'
        ],
    },
    ("mention", "num"): {
        "deck": ["{name} runs its operations on {company}", "Customers include {name}"],
        "press": [
            "{name} recently rolled out {company} across its operations.",
            "{name} selected {company} to replace spreadsheets.",
        ],
    },
    ("factual_partner", "num"): {
        "deck": ["Official integration partner of {partner} since 2025"],
        "press": ["{company} became an official integration partner of {partner} in 2025."],
    },
    ("factual_award", "num"): {
        "deck": ["Named to {award}"],
        "press": ["{company} was named to {award}."],
    },
}

SLIDE_OF = {
    "arr": 7,
    "mrr": 7,
    "revenue": 7,
    "arr_growth": 7,
    "customers_growth": 7,
    "customers": 6,
    "mention": 6,
    "nrr": 8,
    "grr": 8,
    "churn": 8,
    "acv": 8,
    "top10": 8,
    "ooo": 8,
    "forecast": 11,
    "opinion": 4,
    "factual": 12,
}

MEMO = {
    "arr": {
        "vague": "ARR is our annualized recurring revenue run-rate.",
        "d01_disclosed": "ARR includes annualized implementation fees.",
        "d01_hint": "ARR is calculated as total monthly billings multiplied by twelve.",
        "h04_arr": "We define ARR as MRR from active subscriptions multiplied by 12, "
        "excluding one-time and implementation fees.",
    },
    "revenue": {
        "vague": "Partner-sourced revenue is recorded in line with our partner agreements.",
        "direct_only": "All revenue is billed directly to customers; "
        "we do not sell through resellers.",
        "d05_disclosed": "Revenue from reseller partners is reported at the gross amount "
        "billed to end customers.",
        "h04_revenue": "Revenue is all recognized revenue (subscriptions, implementation and "
        "training), net of reseller commissions and credit notes.",
    },
    "concentration": {
        "vague": "",
        "h04_top10_concentration": "Customer concentration is the share of trailing twelve-month "
        "recognized revenue that comes from our ten largest customers.",
    },
    "customers": {
        "vague": "A customer is an organization with a contract with us.",
        "d03_hint": "We count every signed account as a customer, "
        "including accounts in onboarding.",
        "h04_active_customers": "Active customers are live accounts with a paid subscription in "
        "the month; pilots and signed accounts that are not yet live are excluded.",
    },
    "churn": {
        "vague": "Logo churn is the share of customers who leave us in a month.",
        "d07_disclosed": "Logo churn is measured on core accounts "
        "and excludes pilots and {smb} accounts.",
        "h04_logo_churn": "Logo churn is the share of paying customers at the start of a month who "
        "cancel during that month, averaged over the period; all paying customers are included.",
    },
    "nrr": {
        "vague": "Net revenue retention compares recurring revenue "
        "from existing customers year over year.",
        "d06_disclosed": "NRR is measured on retained customers: accounts active at both the start "
        "and the end of the period.",
        "d06_hint": "NRR reflects how much our retained accounts expand over twelve months.",
        "h04_nrr": "NRR is subscription revenue this month from customers "
        "active twelve months ago, "
        "divided by their subscription revenue then; churned customers count as zero.",
        "h04_grr": "GRR is subscription revenue this month from customers "
        "active twelve months ago, "
        "capped at each customer's revenue then, divided by their revenue then; churned customers "
        "count as zero.",
    },
}


def _clean(text: str) -> str:
    return " ".join(text.split()).replace(" .", ".").replace(" ,", ",").strip()


def render_sentence(spec: ClaimSpec, venue: str, company: dict, rng: np.random.Generator) -> str:
    """Текст одного утверждения для дека (буллет) или пресс-релиза (предложение)."""
    if spec.key == "ooo":
        text = spec.template.format(v=spec.value_text)
        if venue == "press":
            body = text if text[:3].isupper() else text[0].lower() + text[1:]
            text = f"The company reports {body}."
        return text
    key = f"factual_{spec.extra['kind']}" if spec.key == "factual" else spec.key
    options = TEMPLATES[(key, spec.form if key.endswith("growth") else "num")][venue]
    if spec.claim_type in {"level", "count"} and spec.period_end != company["end"]:
        options = [o for o in options if "{p}" in o]  # прошлый период обязан быть назван в тексте
    tmpl = (
        spec.template
        if (spec.template and venue == "deck")
        else options[int(rng.integers(len(options)))]
    )
    q = "below " if venue == "press" and spec.q_text.startswith("<") else spec.q_text
    return _clean(
        tmpl.format(
            q=q,
            v=spec.value_text,
            p=spec.period_text,
            company=company["name"],
            noun=company["noun"],
            category=company["category"],
            name=spec.value_text,
            partner=spec.extra.get("partner", ""),
            award=spec.extra.get("award", ""),
        )
    )


def render_deck(company: dict, items: dict[int, list[str]]) -> str:
    """Дек: фиксированная структура из 13 слайдов; items — буллеты-утверждения по слайдам."""
    name, noun, cat = company["name"], company["noun"], company["category"]
    slides = {
        1: (f"# {name}", [f"{cat.capitalize()} for {noun}", "Series A investor presentation"]),
        2: (
            "## The problem",
            [
                f"{noun.capitalize()} still run key workflows in spreadsheets",
                "Errors and manual work eat into thin margins",
            ],
        ),
        3: (
            "## Our solution",
            [f"{name} automates {cat} end to end", "Connects to the tools teams already use"],
        ),
        4: (
            "## Why now",
            [
                "Operators are consolidating their software stack",
                "Cloud adoption in the segment keeps accelerating",
            ],
        ),
        5: ("## Product", None),
        6: ("## Customers", []),
        7: ("## Traction", []),
        8: ("## Retention and unit economics", []),
        9: ("## Recurring revenue by month", None),
        10: (
            "## Go-to-market",
            [
                f"Direct sales to {noun}",
                "Partnerships with channel resellers"
                if company["has_resellers"]
                else "Inbound and outbound sales",
            ],
        ),
        11: ("## Outlook", ["Expanding the product into adjacent workflows"]),
        12: ("## Team and recognition", ["Founders with operating experience in the industry"]),
        13: ("## The ask", ["Raising a Series A to scale sales and product"]),
    }
    parts = []
    for num, (title, bullets) in slides.items():
        if bullets is None:
            body = f"<!-- image-only -->\n![{title[3:]}](images/slide_{num:02d}.png)"
        else:
            body = "\n".join([title, "", *[f"- {b}" for b in [*items.get(num, []), *bullets]]])
            if not items.get(num) and not bullets:
                body = title
        parts.append(f"<!-- slide: {num} -->\n{body}\n")
    return "\n".join(parts)


def render_press_release(company: dict, idx: int, paragraphs: list[str], date: str) -> list[str]:
    """Блоки пресс-релиза (блок 1 — заголовок); утверждения — абзацы `paragraphs`."""
    name, noun, cat = company["name"], company["noun"], company["category"]
    headlines = [
        "shares business update",
        "announces new customer milestones",
        "reports progress across its customer base",
    ]
    return [
        f"# {name} {headlines[(idx - 1) % len(headlines)]}",
        f"{company['city'].upper()}, {date} — {name}, the {cat} platform for {noun}, "
        "today shared an update on its business.",
        *paragraphs,
        f'"Our customers trust us with work they cannot afford to get wrong," '
        f"said the CEO of {name}.",
        f"## About {name}",
        f"{name} builds {cat} software for {noun}. The company is backed by venture investors.",
    ]


def render_memo(company: dict, memo: dict[str, str], first: str, end: str) -> str:
    """Меморандум CFO: 5 страниц, определения метрик «как считает компания»."""
    name = company["name"]

    memo = {"revenue": "vague" if company["has_resellers"] else "direct_only", **memo}

    def d(topic: str) -> str:
        return MEMO[topic][memo.get(topic, "vague")].format(smb=company["smb_label"])

    pages = [
        f"# {name}: CFO memo on metric definitions\n\nPrepared for the Series A data room. "
        f"The billing export covers {first} to {end}. This memo explains how management "
        "computes the metrics used in our investor materials.",
        "## Revenue recognition\n\nSubscription invoices are recognized evenly over the service "
        "period; annual prepayments are spread across twelve months. Invoices in EUR and GBP are "
        "converted at the monthly rate of the first service month. "
        "Voided invoices are excluded and "
        f"credit notes are netted against revenue.\n\n{d('revenue')}",
        f"## ARR and MRR\n\nMRR is reported monthly by the finance team.\n\n{d('arr')}",
        f"## Customers and churn\n\n{d('customers')}\n\n{d('churn')}",
        f"## Retention\n\n{d('nrr')}\n\n"
        "Cohorts are tracked in the cohort workbook in the data room."
        + (f"\n\n{d('concentration')}" if d("concentration") else ""),
    ]
    blocks = (f"<!-- page: {i} -->\n{p}\n" for i, p in enumerate(pages, start=1))
    return "\n".join(blocks)
