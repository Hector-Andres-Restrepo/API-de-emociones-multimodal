"""Interfaces de los motores de inferencia y utilidades de rostro compartidas."""
from dataclasses import dataclass
from typing import Dict, List, Protocol, Tuple

import numpy as np


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


class TextEngine(Protocol):
    ready: bool
    version: str

    def load(self) -> None: ...
    def predict(self, text: str) -> Tuple[Dict[str, float], List[str]]:
        """Devuelve (probabilidades canónicas, advertencias)."""


class VisualEngine(Protocol):
    ready: bool
    version: str
    window: int

    def load(self) -> None: ...
    def detect_faces(self, bgr: np.ndarray) -> List[Box]:
        """Rostros ordenados por área descendente."""
    def embed(self, bgr: np.ndarray, box: Box) -> Tuple[Dict[str, float], np.ndarray]:
        """(probabilidades canónicas por frame [modelo estático], vector de 512 características)."""
    def temporal(self, window: np.ndarray) -> Dict[str, float]:
        """Probabilidades canónicas del LSTM para una ventana (W, 512)."""


def crop_face(bgr: np.ndarray, box: Box) -> np.ndarray:
    h, w = bgr.shape[:2]
    x1, y1 = max(0, box.x1), max(0, box.y1)
    x2, y2 = min(w, box.x2), min(h, box.y2)
    return bgr[y1:y2, x1:x2]
