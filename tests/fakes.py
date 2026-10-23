"""MOCKS SOLO PARA PRUEBAS AUTOMATIZADAS. No se usan en la aplicación real:
la API en ejecución siempre usa los motores reales (RobertuitoEngine / EmoAffectNetEngine)."""
from typing import Dict, List, Tuple

import numpy as np

from app.engines.base import Box
from app.labels import CANONICAL


def onehot(emotion: str, conf: float = 0.9) -> Dict[str, float]:
    rest = (1 - conf) / (len(CANONICAL) - 1)
    return {c: (conf if c == emotion else rest) for c in CANONICAL}


class FakeTextEngine:
    ready, version = True, "fake-text@test"

    def __init__(self, probs=None):
        self.probs = probs or onehot("HAPPINESS")

    def load(self): ...

    def predict(self, text: str) -> Tuple[Dict[str, float], List[str]]:
        return dict(self.probs), []


class KeywordTextEngine:
    """Texto simulado por palabras clave (SOLO PRUEBAS): permite que cambiar el texto cambie la emoción.
    Solo cuenta llamadas; no guarda los textos recibidos."""
    ready, version = True, "fake-keyword-text@test"

    def __init__(self, mapping=None, default=None):
        self.mapping = mapping or {"triste": onehot("SADNESS", 0.8), "feliz": onehot("HAPPINESS", 0.9),
                                   "tranquilo": onehot("NEUTRAL", 0.9), "sorpresa": onehot("SURPRISE", 0.9)}
        self.default = default or onehot("NEUTRAL", 0.6)
        self.n_calls = 0

    def load(self): ...

    def predict(self, text: str) -> Tuple[Dict[str, float], List[str]]:
        self.n_calls += 1
        for key, probs in self.mapping.items():
            if key in text.lower():
                return dict(probs), []
        return dict(self.default), []


class FakeVisualEngine:
    ready, version, window = True, "fake-visual@test", 4

    def __init__(self, faces: int = 1, static=None, temporal=None):
        self.faces = faces
        self.static = static or onehot("HAPPINESS")
        self.temporal_probs = temporal or onehot("HAPPINESS")
        self.window_shapes: List[tuple] = []
        self.calls = 0

    def load(self): ...

    def detect_faces(self, bgr) -> List[Box]:
        return [Box(10 + 40 * i, 10, 40 + 40 * i, 60) for i in range(self.faces)]

    def embed(self, bgr, box):
        self.calls += 1
        return dict(self.static), np.full(512, 0.1 * self.calls, dtype=np.float32)

    def temporal(self, window):
        self.window_shapes.append(window.shape)
        return dict(self.temporal_probs)
