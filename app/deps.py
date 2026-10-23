"""Autenticación Bearer, límite de tasa, compuerta de concurrencia y utilidades comunes."""
import hmac
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict

from fastapi import Request

from .config import Settings
from .errors import ApiError

log = logging.getLogger("emotions.deps")

DISCLAIMER_FACE = "Las expresiones faciales no determinan el estado emocional interno."
DISCLAIMER_TEXT = "La clasificación de texto no determina el estado emocional interno ni la sinceridad de la persona."
DISCLAIMER_INCONGRUENCE = "La incongruencia no permite inferir engaño ni ocultamiento emocional."


class RateLimiter:
    """Ventana deslizante de 60 s por cliente (en memoria, por proceso)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True


def authenticate(request: Request) -> None:
    s: Settings = request.app.state.settings
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token or not hmac.compare_digest(token.encode(), s.api_token.encode()):
        raise ApiError(401, "UNAUTHORIZED", "Token Bearer ausente o inválido.")
    client = request.client.host if request.client else "unknown"
    if not request.app.state.limiter.allow(client):
        raise ApiError(429, "RATE_LIMIT_EXCEEDED", "Demasiadas solicitudes. Reintente en unos segundos.")


def require_text(request: Request):
    e = request.app.state.text
    if not e.ready:
        raise ApiError(503, "MODEL_UNAVAILABLE", "El modelo de texto no está cargado.", {"model": "text", "loadError": request.app.state.load_errors.get("text")})
    return e


def require_visual(request: Request):
    e = request.app.state.visual
    if not e.ready:
        raise ApiError(503, "MODEL_UNAVAILABLE", "El modelo visual no está cargado.", {"model": "visual", "loadError": request.app.state.load_errors.get("visual")})
    return e


def run_inference(state, fn: Callable, *args):
    """Ejecuta `fn` bajo el límite de concurrencia; traduce fallos a INFERENCE_ERROR sin registrar contenido."""
    if not state.gate.acquire(timeout=30):
        raise ApiError(429, "RATE_LIMIT_EXCEEDED", "Servidor ocupado; reintente en unos segundos.")
    try:
        return fn(*args)
    except ApiError:
        raise
    except Exception as e:
        log.error("Fallo de inferencia: %s", type(e).__name__)
        raise ApiError(500, "INFERENCE_ERROR", "Error interno de inferencia.")
    finally:
        state.gate.release()


def validate_text(text, language, settings: Settings) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ApiError(400, "VALIDATION_ERROR", "El campo 'text' es obligatorio y no puede estar vacío.", {"field": "text"})
    if len(text) > settings.max_text_chars:
        raise ApiError(400, "VALIDATION_ERROR", f"El texto supera {settings.max_text_chars} caracteres.", {"field": "text"})
    if (language or "es").lower() != "es":
        raise ApiError(422, "UNSUPPORTED_LANGUAGE", "Solo se admite español ('es') por ahora.", {"field": "language", "supported": ["es"]})
    return text
