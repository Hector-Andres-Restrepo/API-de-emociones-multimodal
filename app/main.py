"""Aplicación FastAPI: API de reconocimiento de emociones multimodal (contrato v1.0.0).

Uso académico. Estima categorías de expresión facial y clasificación de texto; NO determina
la emoción interna, la sinceridad ni el estado psicológico de una persona.
"""
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from .api import emotions, streams
from .api import live as live_api  # (no usar el nombre `live`: lo ocupa el endpoint /health/live)
from .config import Settings
from .deps import RateLimiter
from .errors import install_handlers
from .services.streams import StreamStore
from .services.video import JobStore, cleanup_stale_dirs

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("emotions")

DESCRIPTION = (
    "Análisis de señales emocionales a partir de texto, imagen, video y cámara (WebSocket), con fusión tardía "
    "de probabilidades e indicador de incongruencia.\n\n"
    "**Limitaciones:** el sistema estima categorías de expresiones faciales y clasificaciones de texto; no determina "
    "de forma fiable la emoción interna, la sinceridad ni el estado psicológico. La incongruencia NO indica engaño. "
    "Uso académico; los pesos de Emo-AffectNet se publican solo para uso científico. "
    "No usar para decisiones laborales, disciplinarias, médicas o de alto impacto."
)


def _load_models(app: FastAPI) -> None:
    for name in ("text", "visual"):
        eng = getattr(app.state, name)
        if eng.ready:
            continue
        try:
            eng.load()
            log.info("Modelo %s listo: %s", name, eng.version)
        except Exception as e:
            app.state.load_errors[name] = f"{type(e).__name__}: {str(e)[:300]}"
            log.error("No se pudo cargar el modelo %s: %s", name, app.state.load_errors[name])
            if not app.state.settings.allow_start_without_models:
                raise


def create_app(settings: Optional[Settings] = None, text_engine=None, visual_engine=None, load_models: bool = True) -> FastAPI:
    settings = settings or Settings.from_env()
    if not settings.api_token:
        raise RuntimeError("API_TOKEN no está definido. Configure la variable de entorno (ver .env.example).")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        os.makedirs(settings.tmp_dir, exist_ok=True)
        cleanup_stale_dirs(settings.tmp_dir, max_age_s=0)  # tras un reinicio no debe quedar nada
        if load_models:
            if settings.allow_start_without_models:
                threading.Thread(target=_load_models, args=(app,), daemon=True, name="model-loader").start()
            else:
                _load_models(app)
        yield
        app.state.jobs.shutdown()

    app = FastAPI(title="API de Emociones Multimodal", version="1.0.0", description=DESCRIPTION, lifespan=lifespan)

    if text_engine is None or visual_engine is None:
        from .engines.text_engine import RobertuitoEngine
        from .engines.visual_engine import EmoAffectNetEngine
        text_engine = text_engine or RobertuitoEngine(settings)
        visual_engine = visual_engine or EmoAffectNetEngine(settings)

    app.state.settings = settings
    app.state.text, app.state.visual = text_engine, visual_engine
    app.state.load_errors = {}
    app.state.limiter = RateLimiter(settings.rate_limit_per_minute)
    app.state.gate = threading.BoundedSemaphore(settings.max_concurrent_inferences)
    app.state.jobs = JobStore(settings)
    app.state.streams = StreamStore(settings, visual_engine.window)

    install_handlers(app)

    @app.middleware("http")
    async def request_id_mw(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        resp = await call_next(request)
        resp.headers["X-Request-Id"] = request.state.request_id
        return resp

    app.include_router(emotions.router)
    app.include_router(streams.router)
    app.include_router(streams.ws_router)
    app.include_router(live_api.router)
    app.include_router(live_api.ws_router)

    @app.get("/demo", include_in_schema=False)
    def demo_page():
        """Página de demostración (HTML estático sin secretos). Las llamadas a la API siguen exigiendo Bearer."""
        return FileResponse(os.path.join(os.path.dirname(__file__), "static", "demo.html"), media_type="text/html; charset=utf-8")

    @app.get("/health/live", tags=["health"])
    def live():
        return {"status": "alive"}

    @app.get("/health/ready", tags=["health"])
    def ready():
        models = {"text": bool(app.state.text.ready), "visual": bool(app.state.visual.ready)}
        ok = all(models.values())
        body = {"status": "ready" if ok else "loading_or_unavailable", "models": models,
                "modelVersions": {"text": app.state.text.version, "visual": app.state.visual.version}}
        if app.state.load_errors:
            body["loadErrors"] = app.state.load_errors
        return JSONResponse(body, status_code=200 if ok else 503)

    return app


def _app_factory() -> FastAPI:
    return create_app()


# `uvicorn app.main:app` (ver Dockerfile). Si no hay API_TOKEN, falla con un mensaje claro.
app = _app_factory() if os.getenv("API_TOKEN") else None
if app is None and os.getenv("EMOTIONS_API_NO_AUTOAPP") != "1":
    raise RuntimeError("API_TOKEN no está definido. Copie .env.example a .env y configure un token.")
