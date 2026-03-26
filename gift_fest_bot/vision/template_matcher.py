from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from gift_fest_bot.config import RecognitionConfig
from gift_fest_bot.item_catalog import ITEM_CATALOG


@dataclass(frozen=True)
class RecognitionResult:
    item_name: str | None
    confidence: float
    is_unknown: bool = False
    is_empty: bool = False
    candidates: tuple[tuple[str, float], ...] = ()


@dataclass
class TemplateRecognizer:
    templates_dir: Path
    config: RecognitionConfig
    templates: dict[str, list[np.ndarray]] = field(default_factory=dict)
    template_masks: dict[str, list[np.ndarray]] = field(default_factory=dict)
    descriptors: dict[str, list[tuple[list[cv2.KeyPoint], np.ndarray]]] = field(default_factory=dict)
    empty_templates: list[np.ndarray] = field(default_factory=list)
    empty_masks: list[np.ndarray] = field(default_factory=list)
    all_item_dirs: list[str] = field(default_factory=list)
    expected_item_dirs: set[str] = field(default_factory=set)
    strict_mode_due_to_missing_templates: bool = False

    def __post_init__(self) -> None:
        self.orb = cv2.ORB_create(nfeatures=300)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.expected_item_dirs = {
            name
            for name in ITEM_CATALOG.keys()
            if isinstance(name, str)
            and name.isascii()
            and name == name.lower()
            and name.replace("_", "").isalnum()
        }
        self._load_templates()
        missing_or_empty_expected = self.expected_item_dirs.difference(set(self.templates.keys()))
        self.strict_mode_due_to_missing_templates = (
            self.config.require_full_template_coverage and len(missing_or_empty_expected) > 0
        )

    def _load_templates(self) -> None:
        if not self.templates_dir.exists():
            return

        for item_dir in self.templates_dir.iterdir():
            if not item_dir.is_dir():
                continue
            self.all_item_dirs.append(item_dir.name)
            item_templates: list[np.ndarray] = []
            item_masks: list[np.ndarray] = []
            item_descriptors: list[tuple[list[cv2.KeyPoint], np.ndarray]] = []
            for image_path in item_dir.rglob("*"):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                    continue
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                item_templates.append(image)
                item_masks.append(self._foreground_mask(image))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                keypoints, descriptors = self.orb.detectAndCompute(gray, None)
                if descriptors is not None and len(keypoints) > 0:
                    item_descriptors.append((keypoints, descriptors))
            if item_dir.name == "empty":
                self.empty_templates.extend(item_templates)
                self.empty_masks.extend(item_masks)
                continue
            if item_templates:
                self.templates[item_dir.name] = item_templates
                self.template_masks[item_dir.name] = item_masks
            if item_descriptors:
                self.descriptors[item_dir.name] = item_descriptors

    def template_stats(self) -> dict:
        loaded_item_names = sorted(self.templates.keys())
        declared_item_names = sorted(self.all_item_dirs)
        empty_item_dirs = sorted(name for name in declared_item_names if name not in self.templates and name != "empty")
        missing_expected_dirs = sorted(self.expected_item_dirs.difference(set(declared_item_names)))
        missing_or_empty_expected_dirs = sorted(self.expected_item_dirs.difference(set(loaded_item_names)))
        total_templates = sum(len(images) for images in self.templates.values())
        return {
            "templates_dir": str(self.templates_dir),
            "declared_item_dirs": len(declared_item_names),
            "loaded_item_dirs": len(loaded_item_names),
            "total_item_templates": total_templates,
            "empty_templates_count": len(self.empty_templates),
            "missing_item_dirs": empty_item_dirs,
            "missing_expected_dirs": missing_expected_dirs,
            "missing_or_empty_expected_dirs": missing_or_empty_expected_dirs,
            "strict_mode_due_to_missing_templates": self.strict_mode_due_to_missing_templates,
        }

    def recognize(self, cell_image: np.ndarray, allow_empty: bool = True) -> RecognitionResult:
        if cell_image.size == 0:
            return RecognitionResult(item_name=None, confidence=0.0, is_unknown=True)

        centered_cell = self._center_crop(cell_image)
        empty_score = self._match_empty(centered_cell)
        # Use a conservative empty gate: both checks must agree.
        if allow_empty and empty_score >= self.config.empty_match_threshold and self._looks_empty(centered_cell):
            return RecognitionResult(item_name=None, confidence=max(empty_score, 1.0 if empty_score < 0 else empty_score), is_empty=True)

        score_by_name: dict[str, float] = {}

        for item_name, template_list in self.templates.items():
            mask_list = self.template_masks.get(item_name, [])
            template_score = max(
                self._match_template_multiscale(centered_cell, template, mask_list[index] if index < len(mask_list) else None)
                for index, template in enumerate(template_list)
            )
            orb_score = max(0.0, self._match_orb(centered_cell, item_name))
            score = 0.75 * template_score + 0.25 * orb_score
            score_by_name[item_name] = score

        if not score_by_name:
            return RecognitionResult(item_name=None, confidence=0.0, is_unknown=True)

        ranked = sorted(score_by_name.items(), key=lambda pair: pair[1], reverse=True)
        best_name, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0
        top_candidates = tuple((name, round(float(score), 4)) for name, score in ranked[:3])

        if best_score < self.config.match_threshold:
            return RecognitionResult(
                item_name=None,
                confidence=max(best_score, 0.0),
                is_unknown=True,
                candidates=top_candidates,
            )
        ambiguity_margin = self.config.ambiguity_margin + (0.01 if self.strict_mode_due_to_missing_templates else 0.0)
        if second_score > 0 and best_score - second_score < ambiguity_margin:
            return RecognitionResult(item_name=None, confidence=best_score, is_unknown=True, candidates=top_candidates)
        unknown_threshold = self.config.unknown_threshold + (0.03 if self.strict_mode_due_to_missing_templates else 0.0)
        if best_score < min(0.95, unknown_threshold):
            return RecognitionResult(item_name=None, confidence=best_score, is_unknown=True, candidates=top_candidates)

        return RecognitionResult(item_name=best_name, confidence=best_score, candidates=top_candidates)

    def _looks_empty(self, image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = float(np.var(gray))
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / max(edges.size, 1)
        return variance < 180.0 and edge_density < 0.035

    def _foreground_mask(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        non_white = cv2.inRange(hsv, (0, 25, 20), (180, 255, 255))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        mask = cv2.bitwise_or(non_white, edges)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def _center_crop(self, image: np.ndarray) -> np.ndarray:
        ratio = min(max(self.config.center_crop_ratio, 0.3), 1.0)
        if ratio >= 0.999:
            return image
        height, width = image.shape[:2]
        crop_width = max(1, int(round(width * ratio)))
        crop_height = max(1, int(round(height * ratio)))
        start_x = max(0, (width - crop_width) // 2)
        start_y = max(0, (height - crop_height) // 2)
        return image[start_y:start_y + crop_height, start_x:start_x + crop_width]

    def _match_empty(self, cell_image: np.ndarray) -> float:
        if not self.empty_templates:
            return -1.0
        scores = []
        for index, template in enumerate(self.empty_templates):
            mask = self.empty_masks[index] if index < len(self.empty_masks) else None
            scores.append(self._match_template_multiscale(cell_image, template, mask))
        return max(scores) if scores else -1.0

    def _match_template_multiscale(
        self,
        cell_image: np.ndarray,
        template: np.ndarray,
        template_mask: np.ndarray | None,
    ) -> float:
        best = -1.0
        # Small scale sweep reduces confusion from slight zoom/perspective shifts.
        for scale in (0.88, 0.94, 1.0, 1.06, 1.12):
            scaled_template, scaled_mask = self._scale_template(template, template_mask, scale)
            score = self._match_template(cell_image, scaled_template, scaled_mask)
            if score > best:
                best = score
        return best

    def _scale_template(
        self,
        template: np.ndarray,
        template_mask: np.ndarray | None,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        height, width = template.shape[:2]
        target_w = max(8, int(round(width * scale)))
        target_h = max(8, int(round(height * scale)))
        resized_template = cv2.resize(template, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if template_mask is None:
            return resized_template, None
        resized_mask = cv2.resize(template_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return resized_template, resized_mask

    def _match_template(self, cell_image: np.ndarray, template: np.ndarray, template_mask: np.ndarray | None) -> float:
        centered_template = self._center_crop(template)
        resized = cv2.resize(cell_image, (centered_template.shape[1], centered_template.shape[0]))
        normalized_cell = self._normalize_lighting(resized)
        normalized_template = self._normalize_lighting(centered_template)
        if template_mask is None:
            result = cv2.matchTemplate(normalized_cell, normalized_template, cv2.TM_CCOEFF_NORMED)
            base_score = float(result.max())
            return max(base_score, self._edge_similarity(normalized_cell, normalized_template))

        centered_mask = self._center_crop(template_mask)
        mask = cv2.resize(centered_mask, (centered_template.shape[1], centered_template.shape[0]), interpolation=cv2.INTER_NEAREST)
        cell_mask = self._foreground_mask(resized)
        active = mask > 0
        if np.count_nonzero(active) < 16:
            result = cv2.matchTemplate(normalized_cell, normalized_template, cv2.TM_CCOEFF_NORMED)
            return float(result.max())

        result = cv2.matchTemplate(normalized_cell, normalized_template, cv2.TM_CCORR_NORMED, mask=mask)
        corr_score = float(result.max())

        color_diff = np.mean(np.abs(resized[active].astype(np.float32) - centered_template[active].astype(np.float32))) / 255.0
        color_score = max(0.0, 1.0 - color_diff)

        overlap = np.count_nonzero(np.logical_and(active, cell_mask > 0))
        union = np.count_nonzero(np.logical_or(active, cell_mask > 0))
        shape_score = overlap / max(union, 1)

        edge_score = self._edge_similarity(normalized_cell, normalized_template)
        return 0.45 * corr_score + 0.25 * color_score + 0.15 * shape_score + 0.15 * edge_score

    def _normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _edge_similarity(self, image_a: np.ndarray, image_b: np.ndarray) -> float:
        gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
        edge_a = cv2.Canny(gray_a, 70, 180)
        edge_b = cv2.Canny(gray_b, 70, 180)
        overlap = np.count_nonzero(np.logical_and(edge_a > 0, edge_b > 0))
        union = np.count_nonzero(np.logical_or(edge_a > 0, edge_b > 0))
        if union == 0:
            return 0.0
        return overlap / union

    def _match_orb(self, cell_image: np.ndarray, item_name: str) -> float:
        if item_name not in self.descriptors:
            return -1.0

        gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) == 0:
            return -1.0

        best_count = 0
        for _, template_descriptors in self.descriptors[item_name]:
            matches = self.matcher.knnMatch(descriptors, template_descriptors, k=2)
            good = []
            for pair in matches:
                if len(pair) < 2:
                    continue
                first, second = pair
                if first.distance < 0.75 * second.distance:
                    good.append(first)
            best_count = max(best_count, len(good))

        threshold = max(self.config.orb_good_match_threshold, 1)
        return min(best_count / threshold, 1.0)
