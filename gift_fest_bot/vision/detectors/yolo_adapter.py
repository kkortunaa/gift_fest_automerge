from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


class ObjectDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]:
        ...


class YoloAdapter:
    """
    Placeholder adapter for future YOLO integration.
    Implement loading and prediction with ultralytics or ONNXRuntime.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path

    def detect(self, image: np.ndarray) -> list[Detection]:
        _ = image
        return []
