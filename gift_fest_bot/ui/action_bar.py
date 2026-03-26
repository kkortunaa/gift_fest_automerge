from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from gift_fest_bot.config import ActionUiConfig, Point
from gift_fest_bot.domain import BoardGeometry


ACTIONABLE_STATES = {"sell", "delete", "create"}


@dataclass(frozen=True)
class UiStateDetection:
    state_name: str
    center: Point
    confidence: float
    bbox: tuple[int, int, int, int]

    @property
    def is_actionable(self) -> bool:
        return self.state_name in ACTIONABLE_STATES


@dataclass
class ActionBarDetector:
    config: ActionUiConfig
    templates: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._load_templates(self.config.templates_dir)

    def _load_templates(self, root: Path) -> None:
        if not root.exists():
            return

        for state_name in ("sell", "delete", "create", "no_energy"):
            state_dir = root / state_name
            if not state_dir.exists():
                continue
            images = []
            for image_path in state_dir.glob("*.*"):
                image = cv2.imread(str(image_path))
                if image is not None:
                    images.append(image)
            if images:
                self.templates[state_name] = images

    def detect(self, frame: np.ndarray, board: BoardGeometry) -> UiStateDetection | None:
        panel_bbox = self._panel_bbox(frame, board)
        px1, py1, px2, py2 = panel_bbox
        panel = frame[py1:py2, px1:px2]
        if panel.size == 0:
            return None

        button_bbox_local = self._button_bbox(panel)
        bx1, by1, bx2, by2 = button_bbox_local
        button_crop = panel[by1:by2, bx1:bx2]
        if button_crop.size == 0:
            return None

        state_name, confidence = self._classify_button(button_crop)
        if state_name is None:
            return None

        abs_bbox = (px1 + bx1, py1 + by1, px1 + bx2, py1 + by2)
        center = Point((abs_bbox[0] + abs_bbox[2]) // 2, (abs_bbox[1] + abs_bbox[3]) // 2)
        return UiStateDetection(
            state_name=state_name,
            center=center,
            confidence=confidence,
            bbox=abs_bbox,
        )

    def _panel_bbox(self, frame: np.ndarray, board: BoardGeometry) -> tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        panel_top = board.y + board.height + int(board.height * self.config.top_margin_ratio)
        panel_height = int(board.height * self.config.panel_height_ratio)
        panel_bottom = min(height, panel_top + panel_height)
        return board.x, panel_top, min(width, board.x + board.width), panel_bottom

    def _button_bbox(self, panel: np.ndarray) -> tuple[int, int, int, int]:
        height, width = panel.shape[:2]
        button_width = int(width * self.config.button_width_ratio)
        button_height = int(height * self.config.button_height_ratio)
        x2 = width - int(width * self.config.button_right_margin_ratio)
        x1 = max(0, x2 - button_width)
        y1 = int(height * self.config.button_vertical_margin_ratio)
        y2 = min(height, y1 + button_height)
        return x1, y1, x2, y2

    def _classify_button(self, button_crop: np.ndarray) -> tuple[str | None, float]:
        if self.templates:
            best_name: str | None = None
            best_score = -1.0
            for state_name, template_list in self.templates.items():
                for template in template_list:
                    resized = cv2.resize(button_crop, (template.shape[1], template.shape[0]))
                    result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
                    score = float(result.max())
                    if score > best_score:
                        best_score = score
                        best_name = state_name
            if best_score >= self.config.match_threshold:
                return best_name, best_score

        return self._fallback_classification(button_crop)

    def _fallback_classification(self, button_crop: np.ndarray) -> tuple[str | None, float]:
        hsv = cv2.cvtColor(button_crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(button_crop, cv2.COLOR_BGR2GRAY)

        pink_mask = cv2.inRange(hsv, (145, 70, 110), (179, 255, 255))
        purple_mask = cv2.inRange(hsv, (120, 45, 50), (155, 255, 170))
        green_mask = cv2.inRange(hsv, (35, 25, 70), (95, 255, 255))

        pink_ratio = float(np.count_nonzero(pink_mask)) / max(pink_mask.size, 1)
        purple_ratio = float(np.count_nonzero(purple_mask)) / max(purple_mask.size, 1)
        green_ratio = float(np.count_nonzero(green_mask)) / max(green_mask.size, 1)

        if pink_ratio > 0.38:
            return "create", min(0.72 + pink_ratio * 0.2, 0.9)

        if purple_ratio > 0.42:
            return "no_energy", min(0.72 + purple_ratio * 0.18, 0.9)

        if green_ratio < 0.2:
            return None, 0.0

        _, binary = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
        text_span_ratio = np.count_nonzero(binary.max(axis=0)) / max(binary.shape[1], 1)

        if text_span_ratio > 0.56:
            return "create", min(0.72 + (text_span_ratio - 0.56), 0.88)
        if text_span_ratio > 0.46:
            return "sell", min(0.7 + (text_span_ratio - 0.46), 0.84)
        return "delete", min(0.68 + (0.46 - text_span_ratio), 0.82)
