"""Emo-AffectNet (ElenaRyumina): backbone ResNet50 por frame + LSTM temporal.

Licencia: los pesos se publican "for scientific usage only" (README del repositorio
EMO-AffectNetModel). Uso académico únicamente; no comercial.
"""
import hashlib
import logging
import os
import threading
from typing import Dict, List, Tuple

import numpy as np

from ..config import Settings
from ..labels import VISUAL_LABEL_MAP, VISUAL_OUTPUT_ORDER, align
from .base import Box, crop_face

log = logging.getLogger("emotions.visual")

# Media por canal (BGR) usada por el pipeline oficial; sin división por desviación.
BGR_MEAN = np.array([91.4953, 103.8827, 131.0912], dtype=np.float32)
KEY_REMAP: Dict[str, str] = {}  # {clave_en_pt: clave_en_modelo}, solo si inspect_weights lo exige


def _sha8(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


class EmoAffectNetEngine:
    def __init__(self, settings: Settings):
        self.s = settings
        self.ready = False
        self.window = settings.lstm_window
        self.version = "emo-affectnet@unloaded"
        self._lock = threading.Lock()  # FaceMesh y torch no se comparten entre hilos sin control

    def _fetch(self, filename: str) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.s.visual_repo_id, filename, cache_dir=os.path.join(self.s.models_dir, "hf"))

    def _load_state(self, model, path: str, what: str) -> None:
        import torch
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = {KEY_REMAP.get(k, k): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Pesos de {what} incompatibles con la arquitectura. Faltan: {list(missing)[:5]}; "
                f"inesperadas: {list(unexpected)[:5]}. Ejecute scripts/inspect_weights.py y ajuste KEY_REMAP."
            )
        model.eval()

    def load(self) -> None:
        import mediapipe as mp
        import torch
        from .emo_affectnet_arch import EmoBackbone, EmoLSTM

        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        bb_path = self._fetch(self.s.visual_backbone_file)
        lstm_path = self._fetch(self.s.visual_lstm_file)
        self._torch = torch
        self.backbone, self.lstm = EmoBackbone(), EmoLSTM()
        self._load_state(self.backbone, bb_path, "backbone")
        self._load_state(self.lstm, lstm_path, "LSTM")
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=self.s.max_faces, refine_landmarks=False, min_detection_confidence=0.5
        )
        self.version = (
            f"emo-affectnet@{self.s.visual_repo_id}"
            f"[{self.s.visual_backbone_file}:{_sha8(bb_path)}+{self.s.visual_lstm_file}:{_sha8(lstm_path)}]"
        )
        log.info("Orden de salida visual (adaptador explícito): %s", VISUAL_OUTPUT_ORDER)
        self.ready = True

    def detect_faces(self, bgr: np.ndarray) -> List[Box]:
        import cv2
        h, w = bgr.shape[:2]
        with self._lock:
            res = self._mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        boxes = []
        for lm in res.multi_face_landmarks or []:
            xs = [p.x * w for p in lm.landmark]
            ys = [p.y * h for p in lm.landmark]
            b = Box(max(0, int(min(xs))), max(0, int(min(ys))), min(w, int(max(xs))), min(h, int(max(ys))))
            if b.area > 0:
                boxes.append(b)
        return sorted(boxes, key=lambda b: b.area, reverse=True)

    def embed(self, bgr: np.ndarray, box: Box) -> Tuple[Dict[str, float], np.ndarray]:
        import cv2
        torch = self._torch
        crop_rgb = cv2.cvtColor(crop_face(bgr, box), cv2.COLOR_BGR2RGB)
        face = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_NEAREST)
        x = face[..., ::-1].astype(np.float32) - BGR_MEAN  # RGB -> BGR y resta de media
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]
        with self._lock, torch.inference_mode():
            logits, feat = self.backbone(t)
            probs = torch.softmax(logits[0], dim=-1).tolist()
        return align(probs, VISUAL_OUTPUT_ORDER, VISUAL_LABEL_MAP), feat[0].numpy()

    def temporal(self, window: np.ndarray) -> Dict[str, float]:
        torch = self._torch
        x = torch.from_numpy(window.astype(np.float32))[None]
        with self._lock, torch.inference_mode():
            probs = torch.softmax(self.lstm(x)[0], dim=-1).tolist()
        return align(probs, VISUAL_OUTPUT_ORDER, VISUAL_LABEL_MAP)
