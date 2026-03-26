from __future__ import annotations

from collections import defaultdict

from gift_fest_bot.domain import ActionType, BoardState, Cell, PlannedAction
from gift_fest_bot.item_catalog import DeletePolicy, ITEM_CATALOG, canonicalize_item_name


def item_value(item_name: str | None) -> float:
    if item_name is None:
        return -1.0

    info = ITEM_CATALOG[canonicalize_item_name(item_name)]
    value = float(info.tier * 10)
    value -= info.main_distance * 2.5
    value += info.future_potential * 1.5

    if info.main_distance == 1:
        value += 6.0
    elif info.main_distance == 2:
        value += 3.0

    if info.in_main_chain:
        value += 2.0

    if info.delete_policy == DeletePolicy.NEVER:
        value += 1000.0
    elif info.delete_policy == DeletePolicy.LAST_RESORT:
        value += 150.0
    elif info.delete_policy == DeletePolicy.FLEX:
        value += 20.0
    elif info.delete_policy == DeletePolicy.PREFER:
        value -= 30.0

    return value


class StrategyEngine:
    def __init__(self, min_confidence: float = 0.9) -> None:
        self.min_confidence = min_confidence

    def choose_action(self, board: BoardState) -> PlannedAction:
        if board.unknown_cells():
            return PlannedAction(ActionType.NONE, reason="Board contains unknown cells; skipping actions")

        merge = self._best_merge(board)
        if merge:
            return merge

        if board.empty_cells():
            return PlannedAction(ActionType.SPAWN, reason="No merges available and board has empty cells")

        sell_cell = self._best_sell_candidate(board)
        if sell_cell is not None:
            return PlannedAction(ActionType.SELL, source=sell_cell, reason="Board full and no merges available")

        return PlannedAction(ActionType.NONE, reason="No valid action found")

    def _best_merge(self, board: BoardState) -> PlannedAction | None:
        groups: dict[str, list[Cell]] = defaultdict(list)
        for cell in board.cells:
            if cell.item_name is not None and cell.confidence >= self.min_confidence:
                groups[canonicalize_item_name(cell.item_name)].append(cell)

        candidates: list[tuple[float, PlannedAction]] = []
        for item_name, cells in groups.items():
            if len(cells) < 2:
                continue
            if ITEM_CATALOG[item_name].next_item is None:
                continue

            score = self._merge_score(item_name)
            sorted_cells = sorted(cells, key=lambda cell: (cell.row, cell.col))
            source = sorted_cells[0]
            target = sorted_cells[1]
            action = PlannedAction(
                action_type=ActionType.MERGE,
                source=source,
                target=target,
                reason=f"Merge {item_name} with priority score {score:.2f}",
            )
            candidates.append((score, action))

        if not candidates:
            return None

        candidates.sort(key=lambda pair: pair[0], reverse=True)
        return candidates[0][1]

    def _merge_score(self, item_name: str) -> float:
        info = ITEM_CATALOG[item_name]
        score = info.tier * 20.0
        if info.in_main_chain:
            score += 8.0
        score -= info.main_distance * 4.0
        score += info.future_potential
        return score

    def _best_sell_candidate(self, board: BoardState) -> Cell | None:
        filled = [cell for cell in board.cells if cell.item_name is not None]
        if not filled:
            return None

        allowed = [cell for cell in filled if ITEM_CATALOG[canonicalize_item_name(cell.item_name)].delete_policy != DeletePolicy.NEVER]
        if not allowed:
            return None

        return min(
            allowed,
            key=lambda cell: (
                item_value(cell.item_name),
                cell.confidence,
                cell.row,
                cell.col,
            ),
        )
