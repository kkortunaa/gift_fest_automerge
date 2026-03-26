from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from gift_fest_bot.config import BotConfig, Point
from gift_fest_bot.controller.device import AdbDevice
from gift_fest_bot.controller.desktop import DesktopDevice
from gift_fest_bot.domain import ActionType, BoardGeometry, BoardState, PlannedAction
from gift_fest_bot.item_catalog import ITEM_CATALOG, canonicalize_item_name
from gift_fest_bot.logic.state_machine import BotState, StateMachine
from gift_fest_bot.strategy import StrategyEngine, item_value
from gift_fest_bot.ui.action_bar import ActionBarDetector, UiStateDetection
from gift_fest_bot.ui.item_panel import ItemPanelDetection, ItemPanelDetector
from gift_fest_bot.vision.board_detector import BoardDetector
from gift_fest_bot.vision.template_matcher import TemplateRecognizer


@dataclass
class BotRuntime:
    config: BotConfig
    device: object
    detector: BoardDetector
    strategy: StrategyEngine
    action_bar_detector: ActionBarDetector
    item_panel_detector: ItemPanelDetector
    state_machine: StateMachine
    current_board: BoardState | None = None

    @classmethod
    def create(cls, config: BotConfig) -> "BotRuntime":
        recognizer = TemplateRecognizer(config.templates_dir, config.recognition)
        detector = BoardDetector(recognizer=recognizer, recognition_config=config.recognition)
        if config.control_mode == "adb":
            device = AdbDevice(config=config)
        elif config.control_mode == "desktop":
            device = DesktopDevice(config=config)
        else:
            raise ValueError(f"Unsupported control_mode: {config.control_mode}")
        strategy = StrategyEngine(min_confidence=config.recognition.strict_action_confidence_threshold)
        action_bar_detector = ActionBarDetector(config.action_ui)
        item_panel_detector = ItemPanelDetector(recognizer=recognizer)
        return cls(
            config=config,
            device=device,
            detector=detector,
            strategy=strategy,
            action_bar_detector=action_bar_detector,
            item_panel_detector=item_panel_detector,
            state_machine=StateMachine(),
        )

    def analyze_frame(self) -> tuple[BoardState, PlannedAction, np.ndarray]:
        frame = self.device.capture_screen()
        if self.current_board is None:
            geometry = self.detector.detect_geometry(frame, self.config.board_override)
            board = self.detector.parse_board(frame, geometry)
            if self.config.active_recognition_enabled:
                board = self._actively_scan_board(board, debug_dir=Path("debug") if self.config.active_recognition_debug else None)
            self.current_board = board
        else:
            board = self.current_board
        action = self.strategy.choose_action(board)
        return board, action, frame

    def _actively_scan_board(self, board: BoardState, debug_dir: Path | None = None) -> BoardState:
        scanned_cells = []
        if self.config.active_recognition_focus_click:
            focus_point = Point(
                board.geometry.x + board.geometry.width // 2,
                max(0, board.geometry.y - max(20, board.geometry.height // 3)),
            )
            self.device.tap(focus_point)

        target_cells: set[tuple[int, int]] | None = None
        if self.config.debug_single_scan_cell:
            target_cells = {tuple(self.config.debug_single_scan_cell)}

        for cell in board.cells:
            if target_cells is not None and (cell.row, cell.col) not in target_cells:
                scanned_cells.append(cell)
                continue
            if cell.item_name is None and cell.confidence >= self.config.active_recognition_confidence_skip:
                scanned_cells.append(cell)
                continue

            center = board.geometry.cell_center(cell.row, cell.col)
            before_frame = self.device.capture_screen() if debug_dir else None
            self.device.tap(Point(*center))
            refreshed_frame = self.device.capture_screen()
            detection = self.item_panel_detector.detect(refreshed_frame, board.geometry)

            if debug_dir:
                self._save_active_recognition_debug(
                    debug_dir=debug_dir,
                    cell=cell,
                    click_point=center,
                    before_frame=before_frame,
                    after_frame=refreshed_frame,
                    detection=detection,
                )
            if detection is not None and detection.is_empty_hint:
                scanned_cells.append(
                    type(cell)(
                        row=cell.row,
                        col=cell.col,
                        item_name=None,
                        confidence=detection.confidence,
                        bbox=cell.bbox,
                        is_unknown=False,
                        top_candidates=(),
                    )
                )
                continue
            if detection is None or detection.item_name is None:
                scanned_cells.append(cell)
                continue

            scanned_cells.append(
                type(cell)(
                    row=cell.row,
                    col=cell.col,
                    item_name=detection.item_name,
                    confidence=detection.confidence,
                    bbox=cell.bbox,
                    top_candidates=cell.top_candidates,
                )
            )

        return BoardState(geometry=board.geometry, cells=scanned_cells)

    def _save_active_recognition_debug(
        self,
        debug_dir: Path,
        cell,
        click_point: tuple[int, int],
        before_frame,
        after_frame,
        detection: ItemPanelDetection | None,
    ) -> None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        stem = f"scan_r{cell.row}_c{cell.col}"
        if before_frame is not None:
            cv2.imwrite(str(debug_dir / f"{stem}_before.png"), before_frame)
        cv2.imwrite(str(debug_dir / f"{stem}_after.png"), after_frame)
        payload = {
            "row": cell.row,
            "col": cell.col,
            "click_x": click_point[0],
            "click_y": click_point[1],
            "initial_item": cell.item_name,
            "initial_confidence": cell.confidence,
            "detected_item": detection.item_name if detection else None,
            "detected_confidence": detection.confidence if detection else None,
            "detected_text_item": detection.text_name if detection else None,
            "detected_text_confidence": detection.text_confidence if detection else None,
            "detected_raw_text": detection.raw_text if detection else "",
            "icon_bbox": detection.icon_bbox if detection else None,
            "detected_empty_hint": detection.is_empty_hint if detection else False,
        }
        with (debug_dir / f"{stem}.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def execute_action(self, frame: np.ndarray, board: BoardState, action: PlannedAction) -> UiStateDetection | None:
        if action.action_type == ActionType.MERGE and action.source and action.target:
            start = board.geometry.cell_center(action.source.row, action.source.col)
            end = board.geometry.cell_center(action.target.row, action.target.col)
            self.device.swipe(start, end)
            self.current_board = self._apply_action_to_board(board, action)
            return None

        if action.action_type == ActionType.SPAWN:
            panel_state = self.action_bar_detector.detect(frame, board.geometry)
            if panel_state is not None and panel_state.state_name == "create":
                self.device.tap(panel_state.center)
                self.current_board = self._apply_action_to_board(board, action)
                return panel_state
            if panel_state is not None and panel_state.state_name == "no_energy":
                return panel_state
            self.device.tap(self.config.spawn_button)
            self.current_board = self._apply_action_to_board(board, action)
            return panel_state

        if action.action_type == ActionType.SELL and action.source:
            center = board.geometry.cell_center(action.source.row, action.source.col)
            self.device.tap(Point(*center))
            refreshed_frame = self.device.capture_screen()
            detection = self.action_bar_detector.detect(refreshed_frame, board.geometry)
            if detection is not None and detection.state_name in {"sell", "delete"}:
                self.device.tap(detection.center)
                self.current_board = self._apply_action_to_board(board, action)
                return detection
            self.device.tap(self.config.sell_button)
            self.current_board = self._apply_action_to_board(board, action)
            return None

        return None

    def _apply_action_to_board(self, board: BoardState, action: PlannedAction) -> BoardState:
        if action.action_type == ActionType.MERGE and action.source and action.target:
            merged_name = self._merged_item_name(action.source.item_name)
            updated = board.replace_cell(
                action.source.row,
                action.source.col,
                item_name=None,
                confidence=1.0,
                is_unknown=False,
                top_candidates=(),
            )
            updated = updated.replace_cell(
                action.target.row,
                action.target.col,
                item_name=merged_name,
                confidence=1.0 if merged_name is not None else 0.0,
                is_unknown=merged_name is None,
                top_candidates=(),
            )
            return updated

        if action.action_type == ActionType.SELL and action.source:
            return board.replace_cell(
                action.source.row,
                action.source.col,
                item_name=None,
                confidence=1.0,
                is_unknown=False,
                top_candidates=(),
            )

        if action.action_type == ActionType.SPAWN:
            empty_cells = sorted(board.empty_cells(), key=lambda cell: (cell.row, cell.col))
            if not empty_cells:
                return board
            target = empty_cells[0]
            return board.replace_cell(
                target.row,
                target.col,
                item_name="bear",
                confidence=1.0,
                is_unknown=False,
                top_candidates=(),
            )

        return board

    def _merged_item_name(self, item_name: str | None) -> str | None:
        canonical = canonicalize_item_name(item_name)
        if canonical is None:
            return None
        info = ITEM_CATALOG.get(canonical)
        if info is None:
            return None
        return info.next_item

    def run_once(self, debug_dir: str | Path | None = None, execute: bool = False) -> dict:
        self.state_machine.state = BotState.SCAN
        board, action, frame = self.analyze_frame()
        self.state_machine.transition_after_scan()
        self.state_machine.transition_after_plan()
        panel_state = self.action_bar_detector.detect(frame, board.geometry)
        executed_button: UiStateDetection | None = None
        if execute and action.action_type != ActionType.NONE:
            executed_button = self.execute_action(frame, board, action)
            self.state_machine.transition_after_execute()
            self._wait_until_board_stable(board.geometry)
            self.state_machine.transition_after_stable()

        summary = summarize_board(board, action, panel_state, executed_button)
        if debug_dir:
            self._save_debug(frame, board, action, panel_state, executed_button, Path(debug_dir))
        return summary

    def run_loop(self, debug_dir: str | Path | None = None) -> None:
        debug_path = Path(debug_dir) if debug_dir else None
        while True:
            summary = self.run_once(debug_dir=debug_path, execute=True)
            print(json.dumps(summary, indent=2))
            time.sleep(self.config.loop_delay_seconds)

    def _wait_until_board_stable(self, geometry: BoardGeometry) -> None:
        time.sleep(self.config.post_action_settle_ms / 1000.0)
        previous = self.device.capture_screen()

        for _ in range(self.config.stable_check_max_attempts):
            time.sleep(self.config.stable_check_interval_ms / 1000.0)
            current = self.device.capture_screen()

            x1 = max(0, geometry.x)
            y1 = max(0, geometry.y)
            x2 = min(current.shape[1], geometry.x + geometry.width)
            y2 = min(current.shape[0], geometry.y + geometry.height)
            if x2 <= x1 or y2 <= y1:
                return

            prev_roi = previous[y1:y2, x1:x2]
            curr_roi = current[y1:y2, x1:x2]
            if prev_roi.size == 0 or curr_roi.size == 0:
                return

            diff = cv2.absdiff(prev_roi, curr_roi)
            change_ratio = float(np.mean(diff)) / 255.0
            if change_ratio <= self.config.stable_change_threshold:
                return
            previous = current

    def _save_debug(
        self,
        frame,
        board: BoardState,
        action: PlannedAction,
        panel_state: UiStateDetection | None,
        executed_button: UiStateDetection | None,
        debug_dir: Path,
    ) -> None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cells_dir = debug_dir / "cells"
        cells_dir.mkdir(parents=True, exist_ok=True)
        for existing in cells_dir.glob("*.png"):
            existing.unlink()

        annotated = frame.copy()
        for cell in board.cells:
            x1, y1, x2, y2 = cell.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = cell.item_name or "empty"
            cv2.putText(
                annotated,
                f"{label}:{cell.confidence:.2f}",
                (x1 + 6, y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            crop = frame[y1:y2, x1:x2]
            safe_label = label.replace("/", "_").replace("\\", "_")
            confidence = f"{cell.confidence:.3f}"
            cv2.imwrite(str(cells_dir / f"r{cell.row}_c{cell.col}_{safe_label}_{confidence}.png"), crop)
        if panel_state is not None:
            x1, y1, x2, y2 = panel_state.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 40), 2)
            cv2.putText(
                annotated,
                f"panel:{panel_state.state_name}:{panel_state.confidence:.2f}",
                (x1 + 6, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 180, 40),
                2,
                cv2.LINE_AA,
            )
        if executed_button is not None:
            x1, y1, x2, y2 = executed_button.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 220, 255), 2)
            cv2.putText(
                annotated,
                f"tap:{executed_button.state_name}:{executed_button.confidence:.2f}",
                (x1 + 6, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (40, 220, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(debug_dir / "last_frame.png"), annotated)
        with (debug_dir / "last_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summarize_board(board, action, panel_state, executed_button), handle, indent=2)


def summarize_board(
    board: BoardState,
    action: PlannedAction,
    panel_state: UiStateDetection | None = None,
    executed_button: UiStateDetection | None = None,
) -> dict:
    rows = []
    for row in range(board.geometry.rows):
        rows.append([board.get(row, col).item_name for col in range(board.geometry.cols)])

    cells = []
    for cell in board.cells:
        cells.append(
            {
                "row": cell.row,
                "col": cell.col,
                "item": cell.item_name,
                "confidence": round(cell.confidence, 3),
                "value": round(item_value(cell.item_name), 2) if cell.item_name else None,
                    "top_candidates": [{"item": name, "score": score} for name, score in cell.top_candidates],
            }
        )

    return {
        "board": {
            "x": board.geometry.x,
            "y": board.geometry.y,
            "width": board.geometry.width,
            "height": board.geometry.height,
            "rows": rows,
            "cells": cells,
        },
        "action": {
            "type": action.action_type.value,
            "source": asdict(action.source) if action.source else None,
            "target": asdict(action.target) if action.target else None,
            "reason": action.reason,
            "panel_state": {
                "state_name": panel_state.state_name,
                "confidence": round(panel_state.confidence, 3),
                "center": asdict(panel_state.center),
            }
            if panel_state
            else None,
            "ui_button": {
                "state_name": executed_button.state_name,
                "confidence": round(executed_button.confidence, 3),
                "center": asdict(executed_button.center),
            }
            if executed_button
            else None,
        },
    }
