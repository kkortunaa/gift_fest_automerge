from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from gift_fest_bot.config import RecognitionConfig
from gift_fest_bot.domain import BoardGeometry, BoardState, Cell
from gift_fest_bot.vision.template_matcher import TemplateRecognizer


@dataclass
class BoardDetector:
    recognizer: TemplateRecognizer
    recognition_config: RecognitionConfig

    def detect_geometry(
        self,
        image: np.ndarray,
        override: Optional[tuple[int, int, int, int]] = None,
    ) -> BoardGeometry:
        if override:
            return BoardGeometry(*override)

        white_geometry = self._detect_white_board_region(image)
        if white_geometry is not None:
            return white_geometry

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 140)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = gray.shape[:2]
        image_area = width * height

        best_score = -1.0
        best_rect: tuple[int, int, int, int] | None = None
        target_ratio = 6 / 4

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < width * 0.25 or h < height * 0.15:
                continue

            area = w * h
            if area < image_area * 0.08:
                continue
            if area > image_area * 0.22:
                continue
            if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
                continue

            ratio = w / max(h, 1)
            ratio_penalty = abs(ratio - target_ratio)
            centeredness = 1.0 - (abs((x + w / 2) - width / 2) / (width / 2))
            vertical_bias = 1.0 - abs((y + h / 2) - (height * 0.45)) / max(height * 0.45, 1)
            score = area / image_area + centeredness + vertical_bias - ratio_penalty * 1.5
            if score > best_score:
                best_score = score
                best_rect = (x, y, w, h)

        if best_rect is None:
            height, width = image.shape[:2]
            return BoardGeometry(0, 0, width, height)

        return BoardGeometry(*best_rect)

    def _detect_white_board_region(self, image: np.ndarray) -> BoardGeometry | None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        image_area = width * height

        white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))
        white_mask = cv2.medianBlur(white_mask, 5)
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score = -1.0
        best_rect: tuple[int, int, int, int] | None = None

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < image_area * 0.04 or area > image_area * 0.5:
                continue
            if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
                continue

            ratio = w / max(h, 1)
            if not 1.15 <= ratio <= 1.9:
                continue

            roi = gray[y:y + h, x:x + w]
            if roi.size == 0:
                continue

            brightness = float(np.mean(roi)) / 255.0
            centeredness = 1.0 - abs((x + w / 2) - (width * 0.5)) / max(width * 0.5, 1)
            vertical_bias = 1.0 - abs((y + h / 2) - (height * 0.46)) / max(height * 0.46, 1)
            score = brightness + area / image_area * 2.0 + centeredness + vertical_bias
            if score > best_score:
                best_score = score
                best_rect = (x, y, w, h)

        if best_rect is None:
            return None

        return BoardGeometry(*best_rect)

    def parse_board(self, image: np.ndarray, geometry: BoardGeometry) -> BoardState:
        grid_geometry = self._grid_geometry(geometry)
        cells: list[Cell] = []
        for row in range(grid_geometry.rows):
            for col in range(grid_geometry.cols):
                x1, y1, x2, y2 = grid_geometry.cell_bbox(row, col)
                crop = image[y1:y2, x1:x2]
                margin_x = int((x2 - x1) * self.recognition_config.cell_margin_ratio)
                margin_y = int((y2 - y1) * self.recognition_config.cell_margin_ratio)
                focus = crop[margin_y: max(crop.shape[0] - margin_y, margin_y + 1), margin_x: max(crop.shape[1] - margin_x, margin_x + 1)]
                recognition = self.recognizer.recognize(focus)
                cells.append(
                    Cell(
                        row=row,
                        col=col,
                        item_name=recognition.item_name,
                        confidence=recognition.confidence,
                        bbox=(x1, y1, x2, y2),
                        is_unknown=recognition.is_unknown,
                        top_candidates=recognition.candidates,
                    )
                )
        return BoardState(geometry=grid_geometry, cells=cells)

    def _grid_geometry(self, outer_geometry: BoardGeometry) -> BoardGeometry:
        left = outer_geometry.x + int(round(outer_geometry.width * self.recognition_config.board_left_ratio))
        top = outer_geometry.y + int(round(outer_geometry.height * self.recognition_config.board_top_ratio))
        right = outer_geometry.x + outer_geometry.width - int(round(outer_geometry.width * self.recognition_config.board_right_ratio))
        bottom = outer_geometry.y + outer_geometry.height - int(round(outer_geometry.height * self.recognition_config.board_bottom_ratio))

        width = max(outer_geometry.cols, right - left)
        height = max(outer_geometry.rows, bottom - top)
        return BoardGeometry(left, top, width, height, cols=outer_geometry.cols, rows=outer_geometry.rows)
