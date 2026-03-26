from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ActionType(str, Enum):
    MERGE = "merge"
    SPAWN = "spawn"
    SELL = "sell"
    NONE = "none"


@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    item_name: Optional[str]
    confidence: float
    bbox: tuple[int, int, int, int]
    is_unknown: bool = False
    top_candidates: tuple[tuple[str, float], ...] = ()

    @property
    def is_empty(self) -> bool:
        return self.item_name is None and not self.is_unknown


@dataclass(frozen=True)
class BoardGeometry:
    x: int
    y: int
    width: int
    height: int
    cols: int = 6
    rows: int = 4

    @property
    def cell_width(self) -> float:
        return self.width / self.cols

    @property
    def cell_height(self) -> float:
        return self.height / self.rows

    def cell_bbox(self, row: int, col: int) -> tuple[int, int, int, int]:
        x1 = int(round(self.x + col * self.cell_width))
        y1 = int(round(self.y + row * self.cell_height))
        x2 = int(round(self.x + (col + 1) * self.cell_width))
        y2 = int(round(self.y + (row + 1) * self.cell_height))
        return x1, y1, x2, y2

    def cell_center(self, row: int, col: int) -> tuple[int, int]:
        x1, y1, x2, y2 = self.cell_bbox(row, col)
        return (x1 + x2) // 2, (y1 + y2) // 2


@dataclass
class BoardState:
    geometry: BoardGeometry
    cells: list[Cell] = field(default_factory=list)

    def get(self, row: int, col: int) -> Cell:
        return self.cells[row * self.geometry.cols + col]

    def empty_cells(self) -> list[Cell]:
        return [cell for cell in self.cells if cell.is_empty]

    def unknown_cells(self) -> list[Cell]:
        return [cell for cell in self.cells if cell.is_unknown]

    def replace_cell(
        self,
        row: int,
        col: int,
        *,
        item_name: Optional[str],
        confidence: float,
        is_unknown: bool | None = None,
        top_candidates: tuple[tuple[str, float], ...] | None = None,
    ) -> "BoardState":
        index = row * self.geometry.cols + col
        current = self.cells[index]
        updated = Cell(
            row=row,
            col=col,
            item_name=item_name,
            confidence=confidence,
            bbox=current.bbox,
            is_unknown=current.is_unknown if is_unknown is None else is_unknown,
            top_candidates=current.top_candidates if top_candidates is None else top_candidates,
        )
        cells = list(self.cells)
        cells[index] = updated
        return BoardState(geometry=self.geometry, cells=cells)


@dataclass(frozen=True)
class PlannedAction:
    action_type: ActionType
    source: Optional[Cell] = None
    target: Optional[Cell] = None
    reason: str = ""
