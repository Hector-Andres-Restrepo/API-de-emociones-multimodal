"""Sesiones de cámara en vivo (WebSocket). Token efímero de un solo uso; ventana temporal por sesión."""
import hashlib
import hmac
import secrets
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import numpy as np

from ..config import Settings
from ..errors import ApiError


@dataclass
class StreamSession:
    session_id: str
    sample_fps: float
    token_hash: str
    token_expires: float
    created: float = field(default_factory=time.time)
    token_used: bool = False
    connected: bool = False
    last_frame_at: float = 0.0
    window: Optional[Deque[np.ndarray]] = None
    frames_processed: int = 0
    mode: str = "VISUAL"          # VISUAL (cámara) | MULTIMODAL_LIVE (cámara + texto)
    params: dict = field(default_factory=dict)


def _h(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class StreamStore:
    def __init__(self, settings: Settings, window: int):
        self.s, self.window = settings, window
        self._sessions: Dict[str, StreamSession] = {}
        self._lock = threading.Lock()

    def _purge(self) -> None:
        now = time.time()
        dead = [k for k, v in self._sessions.items()
                if now - v.created > self.s.stream_max_seconds + 30 or (not v.connected and now > v.token_expires)]
        for k in dead:
            del self._sessions[k]

    def create(self, sample_fps: float, mode: str = "VISUAL", params: Optional[dict] = None):
        with self._lock:
            self._purge()
            if len(self._sessions) >= self.s.max_stream_sessions:
                raise ApiError(429, "RATE_LIMIT_EXCEEDED", "Máximo de sesiones de cámara simultáneas alcanzado.")
            token = secrets.token_urlsafe(32)
            sess = StreamSession(f"session-{uuid.uuid4().hex[:12]}", sample_fps, _h(token),
                                 time.time() + self.s.stream_token_ttl_seconds,
                                 window=deque(maxlen=self.window), mode=mode, params=dict(params or {}))
            self._sessions[sess.session_id] = sess
            return sess, token

    def authenticate(self, session_id: str, token: str, mode: str = "VISUAL") -> Optional[StreamSession]:
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess or sess.mode != mode or sess.token_used or time.time() > sess.token_expires:
                return None  # sesión inexistente, de otro tipo, ya usada o expirada
            if not hmac.compare_digest(sess.token_hash, _h(token or "")):
                return None
            sess.token_used, sess.connected = True, True  # un solo uso
            return sess

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
