from gift_fest_bot.bot import BotRuntime
from gift_fest_bot.domain import ActionType, BoardGeometry, BoardState, Cell, PlannedAction
from gift_fest_bot.strategy import StrategyEngine, item_value


def make_board(items):
    geometry = BoardGeometry(0, 0, 600, 400)
    cells = []
    for row in range(4):
        for col in range(6):
            name = items[row][col]
            cells.append(Cell(row=row, col=col, item_name=name, confidence=0.99, bbox=geometry.cell_bbox(row, col)))
    return BoardState(geometry=geometry, cells=cells)


def test_prefers_high_level_merge():
    board = make_board(
        [
            ["bear", "bear", "cake", "cake", None, None],
            [None, None, None, None, None, None],
            [None, None, None, None, None, None],
            [None, None, None, None, None, None],
        ]
    )
    action = StrategyEngine().choose_action(board)
    assert action.source.item_name == "cake"
    assert action.target.item_name == "cake"


def test_spawn_when_empty_and_no_merge():
    board = make_board(
        [
            ["bear", "heart", "gift", None, None, None],
            [None, None, None, None, None, None],
            [None, None, None, None, None, None],
            [None, None, None, None, None, None],
        ]
    )
    action = StrategyEngine().choose_action(board)
    assert action.action_type.value == "spawn"


def test_protects_secondary_progress_items_from_sell():
    assert item_value("seeds") > item_value("thread")
    assert item_value("candles") > item_value("layers")


def test_secondary_items_except_three_allowed_are_never_deleted():
    protected_secondary = [
        "watering_can",
        "seeds",
        "layers",
        "candles",
        "pruner",
        "vase",
        "astronaut",
        "earth",
        "opener",
        "bucket",
        "glasses",
        "whistle",
        "ball",
        "boots",
        "pickaxe",
        "crystal",
        "cart",
        "earrings",
        "crown",
        "necklace",
    ]
    for item_name in protected_secondary:
        assert item_value(item_name) > item_value("bear")


def test_random_nfts_are_never_deleted():
    nft_items = [
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
    for item_name in nft_items:
        assert item_value(item_name) > item_value("bear")


def test_never_sells_protected_main_and_endgame_branches():
    board = make_board(
        [
            ["rose", "cake", "bouquet", "rocket", "champagne", "cup"],
            ["diamond", "ring", "ball", "boots", "pickaxe", "crystal"],
            ["cart", "earrings", "crown", "necklace", "earth", "watering_can"],
            ["vase", "candles", "seeds", "thread", "valentine", "box"],
        ]
    )
    action = StrategyEngine().choose_action(board)
    assert action.action_type.value == "sell"
    assert action.source.item_name in {"thread", "valentine", "box"}


def test_last_resort_main_items_are_kept_until_easy_secondary_trash_is_gone():
    board = make_board(
        [
            ["bear", "heart", "gift", "watering_can", "seeds", "rose"],
            ["layers", "candles", "cake", "pruner", "vase", "bouquet"],
            ["astronaut", "earth", "rocket", "opener", "bucket", "glasses"],
            ["thread", "valentine", "box", "champagne", "diamond", "ring"],
        ]
    )
    action = StrategyEngine().choose_action(board)
    assert action.source.item_name in {"thread", "valentine", "box"}


def test_does_not_merge_terminal_items_without_next_stage():
    board = make_board(
        [
            ["heart_locket_nft", "heart_locket_nft", None, None, None, None],
            [None, None, None, None, None, None],
            [None, None, None, None, None, None],
            [None, None, None, None, None, None],
        ]
    )
    action = StrategyEngine().choose_action(board)
    assert action.action_type.value == "spawn"


def test_runtime_updates_board_after_merge():
    board = make_board(
        [
            ["bear", "bear", None, None, None, None],
            [None, None, None, None, None, None],
            [None, None, None, None, None, None],
            [None, None, None, None, None, None],
        ]
    )
    action = PlannedAction(ActionType.MERGE, source=board.get(0, 0), target=board.get(0, 1))
    runtime = BotRuntime(
        config=None,
        device=None,
        detector=None,
        strategy=None,
        action_bar_detector=None,
        item_panel_detector=None,
        state_machine=None,
    )

    updated = runtime._apply_action_to_board(board, action)

    assert updated.get(0, 0).item_name is None
    assert updated.get(0, 1).item_name == "heart"


def test_runtime_updates_board_after_sell():
    board = make_board(
        [
            ["thread", None, None, None, None, None],
            [None, None, None, None, None, None],
            [None, None, None, None, None, None],
            [None, None, None, None, None, None],
        ]
    )
    action = PlannedAction(ActionType.SELL, source=board.get(0, 0))
    runtime = BotRuntime(
        config=None,
        device=None,
        detector=None,
        strategy=None,
        action_bar_detector=None,
        item_panel_detector=None,
        state_machine=None,
    )

    updated = runtime._apply_action_to_board(board, action)

    assert updated.get(0, 0).item_name is None


def test_runtime_updates_board_after_spawn():
    board = make_board(
        [
            [None, "thread", None, None, None, None],
            [None, None, None, None, None, None],
            [None, None, None, None, None, None],
            [None, None, None, None, None, None],
        ]
    )
    action = PlannedAction(ActionType.SPAWN)
    runtime = BotRuntime(
        config=None,
        device=None,
        detector=None,
        strategy=None,
        action_bar_detector=None,
        item_panel_detector=None,
        state_machine=None,
    )

    updated = runtime._apply_action_to_board(board, action)

    assert updated.get(0, 0).item_name == "bear"
