# DD Verifier

[![lint](https://github.com/gumaonelove/itmo-mlsd/actions/workflows/lint.yml/badge.svg)](https://github.com/gumaonelove/itmo-mlsd/actions/workflows/lint.yml)

**Автоматическая проверка утверждений стартапа по его документам и данным для due diligence венчурного фонда.**

Проект курса ML System Design (ИТМО, магистратура, 2026).

## Проблема

Аналитик венчурного фонда получает питч-дек и пресс-релизы стартапа, который привлекает раунд. Ему нужно понять, где компания говорит правду, где преувеличивает, а где ошибается.

Пример: Pantrio, B2B SaaS для учёта запасов в ресторанах, раунд Series A. Компания заявляет: «ARR вырос в 3 раза». Аналитик проверяет это по базе инвойсов, когортам и меморандуму CFO. Вручную такая проверка занимает дни, поэтому многие утверждения принимаются на веру.

## Решение

DD Verifier принимает документы стартапа и data room и строит **реестр утверждений**:

```text
питч-дек, пресс-релизы ─┐
                         ├─► извлечение утверждений ─► план проверки ─► расчёт по данным ─► вердикт + доказательства
data room (инвойсы,     ─┘        (LLM)                  (LLM)            (SQL, без LLM)        (вердикт: правила P5 в коде;
когорты, меморандум CFO)                                                                         объяснение: LLM-верификатор)
```

По каждому утверждению реестр содержит:

- вердикт: `SUPPORTED`, `EXAGGERATED`, `CONTRADICTED`, `NOT_ENOUGH_DATA` или `UNVERIFIABLE`;
- пересчитанное значение и отклонение от заявленного;
- доказательства: SQL-расчёт и цитаты со ссылкой на страницу;
- черновик вопроса фаундерам (отправляет аналитик).

Числа считает только детерминированный код. LLM отвечает за язык: извлекает утверждения, выбирает метрику и объясняет результат. Финальное решение остаётся за аналитиком.

Подробности: **[дизайн-документ](docs/ml_system_design_doc.md)**.

## Статус

| ДЗ | Содержание | Ветка | Статус |
| --- | --- | --- | --- |
| 1 | Сетап репозитория (ruff, pre-commit, CI); дизайн-документ: цели, требования, скоуп, предпосылки, постановка задачи, блок-схема | `hw_1` | на ревью |
| 2 | EDA (названия KPI в проспектах SEC, реальная выгрузка инвойсов, синтетический data room); онтология метрик, библиотека метрик, генератор синтетики с каталогом расхождений; дизайн-документ v0.4: раздел [2.3 «Этапы решения задачи»](docs/ml_system_design_doc.md#23-этапы-решения-задачи) | `hw_2` | на ревью |
| 3+ | Бейзлайны и MVP, анализ ошибок, внедрение | `hw_n` | — |

Ветка `hw_2` создана от `hw_1`, потому что ДЗ №1 ещё не смержено в `main`.

Где смотреть ДЗ №2:

- сводка EDA и главные находки: [`reports/eda/README.md`](reports/eda/README.md);
- отчёты: [01 — названия KPI в проспектах SEC](reports/eda/01_sec_claims.md), [02 — реальная выгрузка инвойсов как прокси биллинга](reports/eda/02_billing_proxy.md), [03 — синтетический data room](reports/eda/03_synthetic_dataroom.md);
- реальные случаи завышения метрик для каталога: [`docs/real_world_cases.md`](docs/real_world_cases.md);
- карточка данных: [`data/README.md`](data/README.md);
- этапы решения задачи: [дизайн-документ, раздел 2.3](docs/ml_system_design_doc.md#23-этапы-решения-задачи).

Не готово: вторая часть EDA 01 — корпус KPI-предложений из проспектов SEC (квалификаторы, периоды, определения). Частоты терминов и список проспектов уже есть; для скачивания документов архив SEC требует контакт в User-Agent, его задаёт автор (см. ниже).

## Структура репозитория

```text
.
├── docs/
│   ├── ml_system_design_doc.md   # дизайн-документ ML-системы
│   └── real_world_cases.md       # реальные случаи завышения метрик → каталог расхождений
├── configs/
│   ├── ontology.yaml             # онтология метрик: определения, варианты компании, допуски P5
│   ├── discrepancy_catalog.yaml  # каталог посаженных расхождений D01–D11, H01–H04, N01–N02, U01, F01
│   └── synth.yaml                # 15 синтетических компаний (dev / holdout), seed, шум
├── src/dd_verifier/
│   ├── metrics/                  # библиотека метрик (pandas + SQL для DuckDB) и правила допусков
│   ├── data/                     # загрузчики data room и 25 проверок качества
│   └── synth/                    # генератор синтетического data room (CLI: python -m dd_verifier.synth)
├── tests/                        # pytest: метрики на данных с известным ответом, инварианты генератора
├── scripts/                      # скачивание реальных данных: UCI Online Retail II, SEC EDGAR
├── notebooks/                    # ноутбуки EDA (выводы очищаются при коммите)
├── reports/
│   ├── eda/                      # отчёты EDA с выводами
│   └── figures/                  # графики EDA
├── data/
│   ├── README.md                 # карточка данных
│   ├── reference/                # небольшие производные данные для воспроизводимого EDA (в git)
│   ├── synthetic/                # генерируется, в git не попадает
│   └── external/                 # скачивается скриптами, в git не попадает
├── .github/workflows/lint.yml    # CI: pre-commit на каждый пуш
├── .pre-commit-config.yaml       # хуки: ruff, базовые проверки, uv-lock, nbstripout
├── pyproject.toml                # метаданные, зависимости, настройки ruff
├── uv.lock                       # зафиксированные версии зависимостей
├── CONTRIBUTING.md               # как пользоваться линтерами и git-процесс
└── README.md
```

## Быстрый старт для разработки

Нужен [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone git@github.com:gumaonelove/itmo-mlsd.git
cd itmo-mlsd
uv sync                               # окружение + dev-зависимости
uv run pre-commit install             # git-хук перед каждым коммитом
uv run pre-commit run --all-files     # прогнать все проверки вручную
```

Подробнее о линтерах, хуках и git-процессе: [CONTRIBUTING.md](CONTRIBUTING.md).

## Как воспроизвести данные и EDA

Из корня репозитория:

```bash
uv sync
# синтетический data room: 15 компаний за ~4 с → data/synthetic/
uv run python -m dd_verifier.synth --config configs/synth.yaml --out data/synthetic
uv run pytest -q                      # 83 теста: метрики, допуски, загрузчики, генератор
# реальная выгрузка инвойсов UCI Online Retail II (CC BY 4.0): ~2,5 мин → data/external/
uv run python scripts/fetch_online_retail.py
# ноутбуки EDA; графики пишутся в reports/figures/
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_sec_claims.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda_billing_proxy.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_eda_synthetic_dataroom.ipynb
# корпус предложений SEC (вторая часть EDA 01): нужен свой контакт в User-Agent, только через переменную окружения
SEC_USER_AGENT="Имя Фамилия e-mail" uv run python scripts/fetch_sec_claims.py
```

Разработка и EDA читают только `data/synthetic/dev/*` и агрегаты `data/synthetic/holdout_summary.json`. Каталоги holdout-компаний не открываются до финальной оценки.

## Процесс сдачи ДЗ

Каждое ДЗ — отдельная ветка `hw_<n>` и Pull Request в `main`. В reviewers добавляются ментор и два кросс-ревьюера. Мерж после аппрува и зелёного CI.

## Автор

Булат Гумеров · [@gumaonelove](https://github.com/gumaonelove)
