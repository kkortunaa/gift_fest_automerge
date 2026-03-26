from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DeletePolicy(str, Enum):
    NEVER = "never"
    LAST_RESORT = "last_resort"
    FLEX = "flex"
    PREFER = "prefer"


MAIN_CHAIN = [
    "bear",
    "heart",
    "gift",
    "rose",
    "cake",
    "bouquet",
    "rocket",
    "champagne",
    "cup",
    "diamond",
    "ring",
    "nft",
]

NFT_ITEMS = [
    "heart_locket_nft",
    "durovs_cap_nft",
    "precious_peach_nft",
    "heroic_helmet_nft",
    "perfume_bottle_nft",
    "swiss_watch_nft",
    "love_candle_nft",
    "ice_cream_nft",
    "jolly_chimp_nft",
    "snoop_dogg_nft",
]

SECONDARY_CHAINS = [
    ["thread", "bear"],
    ["valentine", "heart"],
    ["box", "gift"],
    ["watering_can", "seeds", "rose"],
    ["layers", "candles", "cake"],
    ["pruner", "vase", "bouquet"],
    ["astronaut", "earth", "rocket"],
    ["opener", "bucket", "glasses", "champagne"],
    ["whistle", "ball", "boots", "cup"],
    ["pickaxe", "crystal", "cart", "diamond"],
    ["earrings", "crown", "necklace", "ring"],
]

DISPLAY_NAMES = {
    "bear": "Мишка",
    "heart": "Сердце",
    "gift": "Подарок",
    "rose": "Роза",
    "cake": "Торт",
    "bouquet": "Букет",
    "rocket": "Ракета",
    "champagne": "Шампанское",
    "cup": "Кубок",
    "diamond": "Алмаз",
    "ring": "Кольцо",
    "nft": "NFT",
    "heart_locket_nft": "Heart Locket",
    "durovs_cap_nft": "Durov's Cap",
    "precious_peach_nft": "Precious Peach",
    "heroic_helmet_nft": "Heroic Helmet",
    "perfume_bottle_nft": "Perfume Bottle",
    "swiss_watch_nft": "Swiss Watch",
    "love_candle_nft": "Love Candle",
    "ice_cream_nft": "Ice Cream",
    "jolly_chimp_nft": "Jolly Chimp",
    "snoop_dogg_nft": "Snoop Dogg",
    "thread": "Нитки",
    "valentine": "Валентинка",
    "box": "Коробка",
    "watering_can": "Лейка",
    "seeds": "Семена",
    "layers": "Коржи",
    "candles": "Свечи",
    "pruner": "Секаторы",
    "vase": "Ваза",
    "astronaut": "Космонавт",
    "earth": "Земля",
    "opener": "Штопор",
    "bucket": "Ведерко",
    "glasses": "Бокалы",
    "whistle": "Свисток",
    "ball": "Мяч",
    "boots": "Бутсы",
    "pickaxe": "Кирка",
    "crystal": "Кристалл",
    "cart": "Ваганетка",
    "earrings": "Серьги",
    "crown": "Корона",
    "necklace": "Ожерелье",
}

NEVER_DELETE_ITEMS = {
    "rose",
    "cake",
    "bouquet",
    "rocket",
    "champagne",
    "cup",
    "diamond",
    "ring",
    "nft",
    "heart_locket_nft",
    "durovs_cap_nft",
    "precious_peach_nft",
    "heroic_helmet_nft",
    "perfume_bottle_nft",
    "swiss_watch_nft",
    "love_candle_nft",
    "ice_cream_nft",
    "jolly_chimp_nft",
    "snoop_dogg_nft",
    "whistle",
    "ball",
    "boots",
    "pickaxe",
    "crystal",
    "cart",
    "earrings",
    "crown",
    "necklace",
}

LAST_RESORT_DELETE_ITEMS = {
    "bear",
    "heart",
    "gift",
}

PREFER_DELETE_ITEMS = {
    "thread",
    "valentine",
    "box",
}

RUSSIAN_NAME_TO_ID = {
    "Мишка": "bear",
    "Сердце": "heart",
    "Подарок": "gift",
    "Роза": "rose",
    "Торт": "cake",
    "Букет": "bouquet",
    "Ракета": "rocket",
    "Шампанское": "champagne",
    "Кубок": "cup",
    "Алмаз": "diamond",
    "Кольцо": "ring",
    "NFT": "nft",
    "Heart Locket": "heart_locket_nft",
    "Durov's Cap": "durovs_cap_nft",
    "Precious Peach": "precious_peach_nft",
    "Heroic Helmet": "heroic_helmet_nft",
    "Perfume Bottle": "perfume_bottle_nft",
    "Swiss Watch": "swiss_watch_nft",
    "Love Candle": "love_candle_nft",
    "Ice Cream": "ice_cream_nft",
    "Jolly Chimp": "jolly_chimp_nft",
    "Snoop Dogg": "snoop_dogg_nft",
    "Нитки": "thread",
    "Валентинка": "valentine",
    "Коробка": "box",
    "Лейка": "watering_can",
    "Семена": "seeds",
    "Коржи": "layers",
    "Свечи": "candles",
    "Секаторы": "pruner",
    "Ваза": "vase",
    "Космонавт": "astronaut",
    "Земля": "earth",
    "Штопор": "opener",
    "Ведерко": "bucket",
    "Бокалы": "glasses",
    "Свисток": "whistle",
    "Мяч": "ball",
    "Бутсы": "boots",
    "Кирка": "pickaxe",
    "Кристалл": "crystal",
    "Ваганетка": "cart",
    "Серьги": "earrings",
    "Корона": "crown",
    "Ожерелье": "necklace",
}


@dataclass(frozen=True)
class ItemInfo:
    name: str
    display_name: str
    tier: int
    main_distance: int
    in_main_chain: bool
    next_item: str | None
    future_potential: int
    delete_policy: DeletePolicy
    notes: str | None = None


def canonicalize_item_name(name: str | None) -> str | None:
    if name is None:
        return None
    return RUSSIAN_NAME_TO_ID.get(name, name)


def _delete_policy_for_item(name: str, main_target: str | None = None) -> DeletePolicy:
    if name in PREFER_DELETE_ITEMS:
        return DeletePolicy.PREFER
    if name in LAST_RESORT_DELETE_ITEMS:
        return DeletePolicy.LAST_RESORT
    if main_target is not None:
        return DeletePolicy.NEVER
    if name in NEVER_DELETE_ITEMS:
        return DeletePolicy.NEVER
    if main_target in {"cup", "diamond", "ring"}:
        return DeletePolicy.NEVER
    return DeletePolicy.FLEX


def build_catalog() -> dict[str, ItemInfo]:
    catalog: dict[str, ItemInfo] = {}

    for index, name in enumerate(MAIN_CHAIN, start=1):
        next_item = MAIN_CHAIN[index] if index < len(MAIN_CHAIN) else None
        catalog[name] = ItemInfo(
            name=name,
            display_name=DISPLAY_NAMES[name],
            tier=index,
            main_distance=0,
            in_main_chain=True,
            next_item=next_item,
            future_potential=len(MAIN_CHAIN) - index,
            delete_policy=_delete_policy_for_item(name),
        )

    nft_tier = catalog["nft"].tier + 1
    nft_future = catalog["nft"].future_potential
    for name in NFT_ITEMS:
        catalog[name] = ItemInfo(
            name=name,
            display_name=DISPLAY_NAMES[name],
            tier=nft_tier,
            main_distance=0,
            in_main_chain=True,
            next_item=None,
            future_potential=nft_future,
            delete_policy=DeletePolicy.NEVER,
        )

    for chain in SECONDARY_CHAINS:
        main_target = chain[-1]
        base_tier = catalog[main_target].tier
        secondary_steps = len(chain) - 1
        for index, name in enumerate(chain[:-1]):
            distance = secondary_steps - index
            notes = None
            catalog[name] = ItemInfo(
                name=name,
                display_name=DISPLAY_NAMES[name],
                tier=max(1, base_tier - distance),
                main_distance=distance,
                in_main_chain=False,
                next_item=chain[index + 1],
                future_potential=catalog[main_target].future_potential + 1,
                delete_policy=_delete_policy_for_item(name, main_target=main_target),
                notes=notes,
            )

    for russian_name, canonical_name in RUSSIAN_NAME_TO_ID.items():
        catalog[russian_name] = catalog[canonical_name]

    return catalog


ITEM_CATALOG = build_catalog()
