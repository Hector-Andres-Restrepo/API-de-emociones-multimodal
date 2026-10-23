"""Análisis temporal de video (Emo-AffectNet: backbone por frame + LSTM sobre ventana de frames)
y gestor de trabajos asíncronos PENDING -> PROCESSING -> COMPLETED/FAILED."""
import logging
import math
import os
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from ..config import Settings
from ..errors import ApiError
from ..labels import CANONICAL, normalize, rounded, top
from .media import remove_dir

log = logging.getLogger("emotions.video")


class JobFailure(Exception):
    def __init__(self, code: str, message: str):
        self.code, self.message = code, message


def probe_video(path: str) -> Tuple[float, float]:
    """(fps, duración en segundos). Lanza ApiError si el archivo no es legible."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ApiError(415, "INVALID_FILE_FORMAT", "No se pudo abrir el video MP4.", {"field": "video"})
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not fps or fps <= 0 or math.isnan(fps):
            fps = 25.0
        if not n or n <= 0:
            raise ApiError(415, "INVALID_FILE_FORMAT", "Video sin fotogramas legibles.", {"field": "video"})
        return float(fps), float(n) / float(fps)
    finally:
        cap.release()


def iter_sampled_frames(path: str, sample_fps: float, max_seconds: Optional[float] = None) -> Iterator[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or math.isnan(fps):
            fps = 25.0
        step, next_t, idx = 1.0 / sample_fps, 0.0, 0
        while cap.grab():
            t = idx / fps
            if max_seconds is not None and t > max_seconds + 1.0:
                raise JobFailure("VALIDATION_ERROR", "El video excede la duración máxima permitida.")
            if t + 1e-9 >= next_t:
                ok, frame = cap.retrieve()
                if ok:
                    yield int(t * 1000), frame
                next_t += step
            idx += 1
    finally:
        cap.release()


@dataclass
class Observation:
    t_ms: int
    faces: int
    probs: Optional[Dict[str, float]]  # salida del LSTM (None si no hubo rostro)


def analyze_video(path: str, sample_fps: float, engine, settings: Settings) -> dict:
    """Procesa secuencias de frames con el LSTM. Ventana deslizante de `engine.window` vectores
    de 512 características; la ventana solo avanza cuando hay rostro (comportamiento del pipeline oficial)."""
    window: deque = deque(maxlen=engine.window)
    obs: List[Observation] = []
    last_t = 0
    for t_ms, frame in iter_sampled_frames(path, sample_fps, settings.max_video_seconds):
        last_t = t_ms
        boxes = engine.detect_faces(frame)
        if not boxes:
            obs.append(Observation(t_ms, 0, None))
            continue
        _, feat = engine.embed(frame, boxes[0])  # rostro principal (mayor área)
        # `feat` ya es ReLU(fc1), la entrada esperada por el LSTM
        if not window:
            window.extend([feat] * engine.window)  # relleno inicial como en el pipeline oficial
        else:
            window.append(feat)
        obs.append(Observation(t_ms, len(boxes), engine.temporal(np.stack(window))))
    return summarize(obs, sample_fps, last_t, settings, engine.window)


def build_timeline(obs: List[Observation], segment_ms: int) -> List[dict]:
    segs: Dict[int, List[Dict[str, float]]] = {}
    for o in obs:
        if o.probs is not None:
            segs.setdefault(o.t_ms // segment_ms, []).append(o.probs)
    timeline = []
    for k in sorted(segs):
        mean = normalize({c: float(np.mean([p[c] for p in segs[k]])) for c in CANONICAL})
        emo, conf = top(mean)
        timeline.append({"startMs": k * segment_ms, "endMs": (k + 1) * segment_ms, "emotion": emo, "confidence": round(conf, 4)})
    return timeline


def summarize(obs: List[Observation], sample_fps: float, last_t_ms: int, settings: Settings, window_frames: int = 0) -> dict:
    with_face = [o for o in obs if o.probs is not None]
    warnings = ["Predicciones visuales sujetas a incertidumbre.", "Las expresiones faciales no determinan el estado emocional interno."]
    result = {
        "durationSeconds": round(last_t_ms / 1000.0, 2),
        "samplingFps": sample_fps,
        "temporalModel": {"type": "LSTM", "windowFrames": window_frames, "framesAnalyzed": len(obs), "framesWithFace": len(with_face)},
        "maxFacesInAFrame": max((o.faces for o in obs), default=0),
        "multiFaceFrames": sum(1 for o in obs if o.faces > 1),
        "dominantExpression": None,
        "timeline": [],
    }
    if not with_face:
        warnings.insert(0, "NO_FACE_DETECTED")
        result["warnings"] = warnings
        return result
    mean = normalize({c: float(np.mean([o.probs[c] for o in with_face])) for c in CANONICAL})
    emo, conf = top(mean)
    result["dominantExpression"] = emo
    result["dominantConfidence"] = round(conf, 4)
    result["probabilities"] = rounded(mean)
    result["timeline"] = build_timeline(obs, int(settings.timeline_segment_seconds * 1000))
    if len(with_face) < len(obs):
        warnings.append("NO_FACE_DETECTED en algunos fotogramas; esos tramos se omiten de la línea de tiempo.")
    result["warnings"] = warnings
    result["_meanProbs"] = mean  # uso interno (fusión); se elimina antes de responder
    return result


def precheck_face_count(path: str, engine, seconds: float = 2.0, sample_fps: float = 2.0) -> int:
    """Comprobación síncrona previa para fusión: máximo de rostros en los primeros segundos."""
    counts = [len(engine.detect_faces(f)) for t, f in iter_sampled_frames(path, sample_fps) if t <= seconds * 1000]
    if not counts:
        return 0
    if any(c > 1 for c in counts):
        return max(counts)
    return 1 if any(c == 1 for c in counts) else 0


# ---------------------------------------------------------------- jobs asíncronos
@dataclass
class Job:
    analysis_id: str
    kind: str  # "video" | "analyze"
    status: str = "PENDING"
    created: float = field(default_factory=time.time)
    finished: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[dict] = None
    inference_ms: Optional[int] = None


class JobStore:
    def __init__(self, settings: Settings):
        self.s = settings
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=settings.max_video_workers, thread_name_prefix="video")

    def _purge(self) -> None:
        now = time.time()
        for k in [k for k, j in self._jobs.items() if j.finished and now - j.finished > self.s.job_retention_seconds]:
            del self._jobs[k]

    def submit(self, kind: str, fn: Callable[[], dict], job_dir: Optional[str]) -> Job:
        with self._lock:
            self._purge()
            active = sum(1 for j in self._jobs.values() if j.status in ("PENDING", "PROCESSING"))
            if active >= self.s.max_video_jobs_queue:
                raise ApiError(429, "RATE_LIMIT_EXCEEDED", "Cola de análisis de video llena. Reintente luego.")
            prefix = "vid" if kind == "video" else "ana"
            job = Job(f"{prefix}-{uuid.uuid4().hex[:8]}", kind)
            self._jobs[job.analysis_id] = job

        def run():
            job.status = "PROCESSING"
            t0 = time.perf_counter()
            try:
                job.result = fn()
                job.status = "COMPLETED"
            except JobFailure as e:
                job.status, job.error = "FAILED", {"code": e.code, "message": e.message}
            except ApiError as e:
                job.status, job.error = "FAILED", {"code": e.code, "message": e.message}
            except Exception as e:  # sin datos del usuario en logs
                log.error("Fallo en job %s: %s", job.analysis_id, type(e).__name__)
                job.status, job.error = "FAILED", {"code": "INFERENCE_ERROR", "message": "Error interno de inferencia."}
            finally:
                job.inference_ms = int((time.perf_counter() - t0) * 1000)
                job.finished = time.time()
                remove_dir(job_dir)  # el video nunca queda en disco

        self._pool.submit(run)
        return job

    def get(self, analysis_id: str, kind: str) -> Optional[Job]:
        with self._lock:
            self._purge()
            j = self._jobs.get(analysis_id)
            return j if j and j.kind == kind else None

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def cleanup_stale_dirs(tmp_dir: str, max_age_s: float) -> int:
    """Elimina directorios job-* huérfanos (p. ej., tras un reinicio)."""
    removed = 0
    if not os.path.isdir(tmp_dir):
        return 0
    now = time.time()
    for name in os.listdir(tmp_dir):
        p = os.path.join(tmp_dir, name)
        if name.startswith("job-") and os.path.isdir(p) and now - os.path.getmtime(p) > max_age_s:
            remove_dir(p)
            removed += 1
    return removed
