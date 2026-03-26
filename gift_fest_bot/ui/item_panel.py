from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

from gift_fest_bot.domain import BoardGeometry
from gift_fest_bot.item_catalog import ITEM_CATALOG, canonicalize_item_name
from gift_fest_bot.vision.template_matcher import TemplateRecognizer

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None

EMPTY_PANEL_HINTS = (
    "соединяй одинаковые предметы и получай более крутые",
    "соединяй одинаковые предметы",
    "получай более крутые",
)


@dataclass(frozen=True)
class ItemPanelDetection:
    item_name: str | None
    confidence: float
    icon_bbox: tuple[int, int, int, int]
    is_empty_hint: bool = False
    text_name: str | None = None
    text_confidence: float = 0.0
    raw_text: str = ""


@dataclass
class ItemPanelDetector:
    recognizer: TemplateRecognizer
    _tesseract_checked: bool = field(default=False, init=False, repr=False)
    _tesseract_ready: bool = field(default=False, init=False, repr=False)
    _tesseract_config_suffix: str = field(default="", init=False, repr=False)

    def detect(self, frame: np.ndarray, grid_geometry: BoardGeometry) -> ItemPanelDetection | None:
        panel_bbox = self.panel_bbox(frame, grid_geometry)
        if panel_bbox is None:
            return None
        panel_left, panel_top, panel_right, panel_bottom = panel_bbox
        panel = frame[panel_top:panel_bottom, panel_left:panel_right]
        if panel.size == 0:
            return None

        icon_x1 = int(panel.shape[1] * 0.02)
        icon_y1 = int(panel.shape[0] * 0.08)
        icon_x2 = int(panel.shape[1] * 0.22)
        icon_y2 = int(panel.shape[0] * 0.92)
        icon = panel[icon_y1:icon_y2, icon_x1:icon_x2]
        if icon.size == 0:
            return None

        text_name, text_confidence, raw_text, is_empty_hint = self._read_item_name(panel)
        item_name = text_name
        confidence = text_confidence

        return ItemPanelDetection(
            item_name=item_name,
            confidence=confidence,
            icon_bbox=(panel_left + icon_x1, panel_top + icon_y1, panel_left + icon_x2, panel_top + icon_y2),
            is_empty_hint=is_empty_hint,
            text_name=text_name,
            text_confidence=text_confidence,
            raw_text=raw_text,
        )

    def panel_bbox(self, frame: np.ndarray, grid_geometry: BoardGeometry) -> tuple[int, int, int, int] | None:
        height, width = frame.shape[:2]
        panel_top = min(height, grid_geometry.y + grid_geometry.height + int(grid_geometry.height * 0.10))
        panel_bottom = min(height, panel_top + int(grid_geometry.height * 0.38))
        panel_left = max(0, grid_geometry.x)
        panel_right = min(width, grid_geometry.x + grid_geometry.width)
        if panel_bottom <= panel_top or panel_right <= panel_left:
            return None
        return panel_left, panel_top, panel_right, panel_bottom

    def panel_crop(self, frame: np.ndarray, grid_geometry: BoardGeometry) -> np.ndarray | None:
        bbox = self.panel_bbox(frame, grid_geometry)
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        panel = frame[y1:y2, x1:x2]
        if panel.size == 0:
            return None
        return panel

    def _read_item_name(self, panel: np.ndarray) -> tuple[str | None, float, str, bool]:
        if pytesseract is None or not self._ensure_tesseract():
            return None, 0.0, "", False
        best_name: str | None = None
        best_item_confidence = 0.0
        best_empty_confidence = 0.0
        best_raw = ""

        for crop in self._iter_text_crops(panel):
            for prepared in self._prepare_ocr_variants(crop):
                for psm in (7, 6):
                    raw = pytesseract.image_to_string(prepared, config=self._tesseract_config(psm))
                    if len(raw.strip()) > len(best_raw.strip()):
                        best_raw = raw
                    for normalized in self._normalized_candidates(raw):
                        empty_confidence = self._match_empty_hint(normalized)
                        if empty_confidence > best_empty_confidence:
                            best_empty_confidence = empty_confidence
                            best_raw = raw
                        match, confidence = self._match_catalog_name(normalized)
                        if match is None or confidence <= best_item_confidence:
                            continue

                        best_name = match
                        best_item_confidence = confidence
                        best_raw = raw

        if best_empty_confidence >= 0.74 and best_empty_confidence >= best_item_confidence:
            return None, best_empty_confidence, best_raw, True
        return best_name, best_item_confidence, best_raw, False

    def _normalize_text(self, text: str) -> str:
        cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
        return " ".join(cleaned.split())

    def _normalize_title_text(self, text: str) -> str:
        normalized = self._normalize_text(text)
        if not normalized:
            return ""

        tokens = []
        for token in normalized.split():
            if token in {"дает", "dayet", "daet"}:
                break
            if any(ch.isdigit() for ch in token):
                break
            tokens.append(token)

        while tokens and len(tokens[0]) == 1:
            tokens = tokens[1:]
        while tokens and len(tokens[-1]) == 1:
            tokens = tokens[:-1]

        candidate = " ".join(tokens)
        alpha_count = sum(ch.isalpha() for ch in candidate)
        if alpha_count < 4:
            return ""
        return candidate

    def _normalized_candidates(self, raw: str) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        for chunk in [*raw.splitlines(), raw]:
            normalized = self._normalize_title_text(chunk)
            if not normalized or normalized in seen:
                if not normalized:
                    continue
            tokens = normalized.split()
            for end in range(1, len(tokens) + 1):
                candidate = " ".join(tokens[:end])
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                candidates.append(candidate)
        return candidates

    def _match_catalog_name(self, normalized_text: str) -> tuple[str | None, float]:
        best_name: str | None = None
        best_score = 0.0
        for alias in ITEM_CATALOG.keys():
            if not isinstance(alias, str):
                continue
            alias_norm = self._normalize_text(alias)
            if not alias_norm:
                continue
            score = self._alias_similarity(alias_norm, normalized_text)
            if score < 0.72 or score <= best_score:
                continue
            canonical = canonicalize_item_name(alias)
            if canonical is None:
                continue
            best_name = canonical
            best_score = score
        return best_name, best_score

    def _match_empty_hint(self, normalized_text: str) -> float:
        best_score = 0.0
        for hint in EMPTY_PANEL_HINTS:
            best_score = max(best_score, self._alias_similarity(hint, normalized_text))
        return best_score

    def _alias_similarity(self, alias_norm: str, normalized_text: str) -> float:
        if alias_norm == normalized_text:
            return 0.99

        if alias_norm.startswith(normalized_text) or normalized_text.startswith(alias_norm):
            shorter = min(len(alias_norm), len(normalized_text))
            longer = max(len(alias_norm), len(normalized_text))
            return 0.9 * (shorter / longer)

        if alias_norm in normalized_text or normalized_text in alias_norm:
            shorter = min(len(alias_norm), len(normalized_text))
            longer = max(len(alias_norm), len(normalized_text))
            return 0.82 * (shorter / longer)

        return SequenceMatcher(a=alias_norm, b=normalized_text).ratio()

    def _iter_text_crops(self, panel: np.ndarray) -> list[np.ndarray]:
        h, w = panel.shape[:2]
        crops = [
            panel[int(h * 0.18) : int(h * 0.78), int(w * 0.14) : int(w * 0.72)],
            panel[int(h * 0.16) : int(h * 0.72), int(w * 0.12) : int(w * 0.74)],
            panel[int(h * 0.12) : int(h * 0.86), int(w * 0.12) : int(w * 0.76)],
        ]
        return [crop for crop in crops if crop.size > 0]

    def _prepare_ocr_variants(self, crop: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            12,
        )
        scaled_binary = cv2.resize(binary, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
        scaled_adaptive = cv2.resize(adaptive, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
        return [scaled_binary, scaled_adaptive]

    def _ensure_tesseract(self) -> bool:
        if self._tesseract_checked:
            return self._tesseract_ready

        self._tesseract_checked = True
        command = shutil.which("tesseract")
        candidates = [Path(command)] if command else []
        candidates.extend(
            [
                Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
                Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
            ]
        )
        executable = next((path for path in candidates if path.exists()), None)
        if executable is None:
            return False

        pytesseract.pytesseract.tesseract_cmd = str(executable)
        project_tessdata = Path(__file__).resolve().parents[2] / "tessdata"
        if project_tessdata.exists():
            self._tesseract_config_suffix = f" --tessdata-dir {project_tessdata}"

        try:
            pytesseract.get_tesseract_version()
        except Exception:
            return False

        self._tesseract_ready = True
        return True

    def _tesseract_config(self, psm: int) -> str:
        return f"--oem 3 --psm {psm} -l rus+eng{self._tesseract_config_suffix}"
