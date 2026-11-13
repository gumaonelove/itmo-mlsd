"""Словари для вымышленных имён клиентов, партнёров и наград (совпадения с реальными случайны)."""

from __future__ import annotations

import numpy as np

from dd_verifier.metrics.customers import normalize_name

FIRST = (  # noqa: SIM905 — компактная запись словаря
    "Amber Ash Birch Blue Bright Cedar Clover Copper Coral Crescent Elm Ember Fern Fox Glen "
    "Golden Granite Harbor Hazel Heron Iron Ivy Juniper Lake Linden Maple Meadow Moss North Oak "
    "Olive Pine Quarry Raven Red River Rowan Sage Silver Slate Spruce Stone Summit Thistle West "
    "Willow Wren Yarrow"
).split()

SECOND = {
    industry: words.split("|")
    for industry, words in {
        "restaurant": "Fork|Ladle|Table|Kitchen|Bistro|Plate|Spoon|Grill|Oven|Tavern|Diner|Eatery",
        "accounting": "Ledger|Advisory|Tax|Audit|Books|Accounting|Finance|Bookkeeping|Counsel|"
        "Numbers|Fiscal|Accounts",
        "logistics": "Freight|Haulage|Transport|Cargo|Logistics|Lines|Couriers|Movers|Express|"
        "Shipping|Carriers|Depot",
        "healthcare": "Health|Clinic|Dental|Care|Medical|Physio|Wellness|Family Practice|"
        "Pediatrics|Vision|Therapy|Surgery",
        "brewery": "Brewing|Brewery|Ales|Hops|Cellars|Malt|Taproom|Craft Beer|Barrel|Cider|Lager|"
        "Distillery",
        "retail": "Market|Goods|Outfitters|Store|Supply|Mercantile|Trading|Boutique|Home|Hardware|"
        "Grocers|Emporium",
        "construction": "Builders|Construction|Contractors|Roofing|Masonry|Homes|Renovation|"
        "Joinery|Civil|Plumbing|Electric|Paving",
        "education": "Academy|School|Learning|Tutors|Institute|College|Prep|Montessori|"
        "Language Center|Studies|Classroom|Campus",
    }.items()
}

DESCRIPTORS = ["", "", "", "Group", "Holdings", "Partners", "Collective"]
LEGAL = {
    "US": ["Inc.", "LLC", "Inc.", ""],
    "GB": ["Ltd", "Limited", ""],
    "EU": ["GmbH", "B.V.", "SAS", ""],
}

PARTNERS = (  # noqa: SIM905
    "Kassaro POS|Ordervale|Tillwise|Ledgerline|Freightory|Medisync Alliance|Hopsheet|Shelfstack|"
    "Brickmark|Classloop|Paylantic|Stackyard Cloud"
).split("|")
AWARDS = [
    "the 2025 Future Ops 50 list",
    "the 2025 Vertical SaaS Awards shortlist",
    "the Regional Tech Growth Awards 2025",
    "the 2025 Operators' Choice list",
]


def name_pool(industry: str, n: int, rng: np.random.Generator) -> list[str]:
    """n уникальных (после нормализации) имён организаций для отрасли."""
    second = SECOND[industry]
    combos = [(a, b, d) for a in FIRST for b in second for d in DESCRIPTORS[3:]]
    combos = [(a, b, "") for a in FIRST for b in second] + combos
    order = rng.permutation(len(combos))
    names, seen = [], set()
    for i in order:
        a, b, d = combos[i]
        name = " ".join(x for x in (a, b, d) if x)
        key = normalize_name(name)
        if key not in seen:
            seen.add(key)
            names.append(name)
        if len(names) == n:
            return names
    raise ValueError(f"не хватает имён для {n} клиентов отрасли {industry}")


def legal_name(base: str, country: str, rng: np.random.Generator) -> str:
    """Добавить юридическую форму по стране."""
    region = country if country in LEGAL else "EU"
    suffix = LEGAL[region][int(rng.integers(len(LEGAL[region])))]
    return f"{base} {suffix}".strip()


def name_variant(name: str, rng: np.random.Generator) -> str:
    """Написание того же имени «как в документах»: без юр. формы, другой регистр, & вместо and."""
    tokens = name.split()
    if normalize_name(tokens[-1]) == "" and len(tokens) > 1:
        tokens = tokens[:-1]
    kind = int(rng.integers(4))
    text = " ".join(tokens)
    if kind == 1:
        return text.upper()
    if kind == 2:
        return f"{text}, Inc." if not text.endswith("Inc.") else text
    if kind == 3:
        return text.lower().title()
    return text
