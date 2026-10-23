"""Sesión multimodal en vivo: fusiona la expresión facial de la cámara con el texto que se escribe.

La lógica vive aquí, separada del transporte (WebSocket), con reloj inyectable para probarla de forma
determinista. Reglas de fusión e incongruencia: las mismas de `app/fusion.py`.

Privacidad: del texto solo se conservan probabilidades, su longitud y un resumen con clave aleatoria por sesión
(para no re-clasificar un texto idéntico). El texto no se escribe en disco ni en logs.
"""
import hashlib
import logging
import math
import os
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from ..config import Settings
from ..deps import DISCLAIMER_FACE, DISCLAIMER_INCONGRUENCE, DISCLAIMER_TEXT
from ..errors import ApiError
from ..fusion import fuse_full, valence_of
from ..labels import CANONICAL, normalize, rounded, top

log = logging.getLogger("emotions.live")

PARTIAL_TEXT_NOTE = "Un texto a medio escribir puede clasificarse distinto del texto final."
EXPERIMENTAL_NOTE = "Pesos, umbral y valencias provienen de una propuesta experimental, no de una regla validada."
FRAME_RATE_TOLERANCE = 0.8  # igual que la cámara: se admite un 20 % de margen sobre sampleFps

STATUS_OK = "OK"
STATUS_WAITING_TEXT = "WAITING_FOR_TEXT"
STATUS_WAITING_FACE = "WAITING_FOR_FACE"
STATUS_FACE_COUNT = "FACE_COUNT_NOT_SUPPORTED_FOR_FUSION"


class LiveAnalyzer:
    def __init__(self, text_engine, visual_engine, settings: Settings, *, sample_fps: float, visual_weight: float,
                 visual_window_s: float, clock: Callable[[], float] = time.monotonic):
        self.text_engine, self.visual = text_engine, visual_engine
        self.s = settings
        self.sample_fps = sample_fps
        self.visual_weight = visual_weight
        self.window_s = visual_window_s
        self.clock = clock
        self.started = clock()

        # estado visual: ventana de características para el LSTM y observaciones recientes (ya clasificadas)
        self._features: Deque[np.ndarray] = deque(maxlen=visual_engine.window)
        self._obs: Deque[Tuple[float, Dict[str, float]]] = deque(maxlen=int(math.ceil(30 * 5)) + 5)
        self._last_frame_t: Optional[float] = None
        self._last_faces: Optional[int] = None

        # estado textual: solo derivados (nunca el texto)
        self._text_probs: Optional[Dict[str, float]] = None
        self._text_warnings: List[str] = []
        self._text_chars = 0
        self._text_version: Optional[int] = None
        self._text_key = os.urandom(16)
        self._text_digest: Optional[bytes] = None

        # control de cadencia
        self._last_frame_accept: float = 0.0
        self._last_text_accept: float = 0.0
        self._accepted_first_frame = False

        # contadores (sin contenido)
        self.frames_processed = 0
        self.texts_analyzed = 0
        self.fusions_computed = 0
        self.incongruent_fusions = 0

    # ------------------------------------------------------------------ cadencia
    def check_frame_rate(self) -> bool:
        """True si el fotograma puede procesarse; False si llega más rápido que sampleFps."""
        now = self.clock()
        if self._accepted_first_frame and now - self._last_frame_accept < FRAME_RATE_TOLERANCE / self.sample_fps:
            return False
        self._last_frame_accept, self._accepted_first_frame = now, True
        return True

    def check_text_rate(self) -> int:
        """0 si el texto puede procesarse; si no, milisegundos que el cliente debe esperar."""
        now = self.clock()
        min_s = self.s.live_text_min_interval_ms / 1000.0
        if self._last_text_accept and now - self._last_text_accept < min_s:
            return max(1, int(math.ceil((min_s - (now - self._last_text_accept)) * 1000)))
        self._last_text_accept = now
        return 0

    # ------------------------------------------------------------------ eventos
    def on_frame(self, bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()
        now = self.clock()
        boxes = self.visual.detect_faces(bgr)
        self._last_frame_t, self._last_faces = now, len(boxes)
        if len(boxes) == 1:  # solo con exactamente un rostro se alimenta el modelo temporal
            _, feat = self.visual.embed(bgr, boxes[0])
            if not self._features:
                self._features.extend([feat] * self.visual.window)  # relleno inicial, como el pipeline oficial
            else:
                self._features.append(feat)
            probs = self.visual.temporal(np.stack(self._features))
            self._obs.append((now, normalize(probs)))
        self.frames_processed += 1
        return self._snapshot("FRAME", t0)

    def on_text(self, text: str, version: Optional[int] = None) -> dict:
        t0 = time.perf_counter()
        if not isinstance(text, str):
            raise ApiError(400, "VALIDATION_ERROR", "El campo 'text' debe ser una cadena.", {"field": "text"})
        if len(text) > self.s.max_text_chars:
            raise ApiError(400, "VALIDATION_ERROR", f"El texto supera {self.s.max_text_chars} caracteres.", {"field": "text"})
        self._text_version = version
        if not text.strip():
            self._text_probs, self._text_warnings, self._text_chars, self._text_digest = None, [], 0, None
            return self._snapshot("TEXT", t0)
        digest = hashlib.blake2b(text.encode("utf-8"), key=self._text_key, digest_size=16).digest()
        if digest != self._text_digest:  # texto nuevo: se clasifica; idéntico: se reutiliza el resultado
            probs, warns = self.text_engine.predict(text)
            self._text_probs, self._text_warnings = normalize(probs), list(warns)
            self._text_digest, self.texts_analyzed = digest, self.texts_analyzed + 1
        self._text_chars = len(text)
        return self._snapshot("TEXT", t0)

    def set_visual_weight(self, weight) -> dict:
        t0 = time.perf_counter()
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0.0 <= float(weight) <= 1.0:
            raise ApiError(400, "VALIDATION_ERROR", "visualWeight debe ser un número entre 0 y 1.", {"field": "visualWeight"})
        self.visual_weight = float(weight)
        return self._snapshot("CONFIG", t0)

    # ------------------------------------------------------------------ estado
    def _recent(self, now: float) -> List[Dict[str, float]]:
        while self._obs and now - self._obs[0][0] > self.window_s:
            self._obs.popleft()
        return [p for _, p in self._obs]

    def _snapshot(self, trigger: str, t0: float) -> dict:
        now = self.clock()
        recent = self._recent(now)
        warnings: List[str] = []

        text = None
        if self._text_probs is not None:
            emo, conf = top(self._text_probs)
            text = {"emotion": emo, "valence": valence_of(emo), "confidence": round(conf, 4),
                    "probabilities": rounded(self._text_probs), "textVersion": self._text_version, "chars": self._text_chars}
            warnings.extend(self._text_warnings)

        visual = None
        mean = None
        if recent:
            mean = normalize({c: float(np.mean([p[c] for p in recent])) for c in CANONICAL})
            emo, conf = top(mean)
            visual = {"emotion": emo, "valence": valence_of(emo), "confidence": round(conf, 4), "faceId": "face-1",
                      "mode": "LSTM", "probabilities": rounded(mean), "framesUsed": len(recent),
                      "windowSeconds": self.window_s,
                      "lastFrameAgeMs": int((now - self._last_frame_t) * 1000) if self._last_frame_t is not None else None,
                      "temporal": {"windowFrames": self.visual.window, "windowFilled": len(self._features)}}

        if self._last_faces is not None and self._last_faces > 1:
            warnings.append(f"MULTIPLE_FACES_DETECTED: {self._last_faces} rostros en el último fotograma; la fusión exige uno solo.")
        if self._last_faces == 0:
            warnings.append("NO_FACE_DETECTED")

        fusion, status = None, STATUS_OK
        if text is None:
            status = STATUS_WAITING_TEXT
        elif self._last_faces is not None and self._last_faces > 1:
            status = STATUS_FACE_COUNT
        elif mean is None:
            status = STATUS_WAITING_FACE
        else:
            f = fuse_full(self._text_probs, mean, self.visual_weight, self.s.incongruence_threshold)
            fusion = {"method": "WEIGHTED_LATE_FUSION", "visualWeight": self.visual_weight,
                      "textWeight": round(1.0 - self.visual_weight, 6), "emotion": f["emotion"],
                      "confidence": round(f["confidence"], 4), "probabilities": rounded(f["probabilities"]),
                      "incongruence": f["incongruence"], "incongruenceThreshold": self.s.incongruence_threshold}
            self.fusions_computed += 1
            self.incongruent_fusions += 1 if f["incongruence"] else 0
            warnings.extend([DISCLAIMER_INCONGRUENCE, EXPERIMENTAL_NOTE])

        if visual is not None:
            warnings.append(DISCLAIMER_FACE)
        if text is not None:
            warnings.extend([DISCLAIMER_TEXT, PARTIAL_TEXT_NOTE])
        warnings = list(dict.fromkeys(warnings))  # sin repetidos, conservando el orden

        return {"type": "update", "trigger": trigger, "timestampMs": int((now - self.started) * 1000),
                "facesDetected": self._last_faces, "text": text, "visual": visual, "fusion": fusion,
                "fusionStatus": status, "inferenceMs": int((time.perf_counter() - t0) * 1000),
                "warnings": warnings}

    def summary(self) -> dict:
        return {"framesProcessed": self.frames_processed, "textsAnalyzed": self.texts_analyzed,
                "fusionsComputed": self.fusions_computed, "incongruentFusions": self.incongruent_fusions}
