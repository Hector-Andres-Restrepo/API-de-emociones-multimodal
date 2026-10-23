"""Configuración por variables de entorno (ver .env.example)."""
import os
from dataclasses import dataclass, replace


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    port: int = 8000
    api_token: str = ""
    models_dir: str = "/models"
    tmp_dir: str = "/tmp/emotions"

    text_model_id: str = "pysentimiento/robertuito-emotion-analysis"
    visual_repo_id: str = "ElenaRyumina/face_emotion_recognition"
    visual_backbone_file: str = "FER_static_ResNet50_AffectNet.pt"
    visual_lstm_file: str = "FER_dinamic_LSTM_Aff-Wild2.pt"
    lstm_window: int = 10
    allow_start_without_models: bool = True

    default_visual_weight: float = 0.5
    incongruence_threshold: float = 0.70

    max_image_bytes: int = 5 * 1024 * 1024
    max_video_bytes: int = 50 * 1024 * 1024
    max_video_seconds: float = 60.0
    max_text_chars: int = 5000
    max_frame_bytes: int = 2_000_000
    max_image_pixels: int = 25_000_000
    max_faces: int = 5
    max_concurrent_inferences: int = 2
    max_video_jobs_queue: int = 10
    max_video_workers: int = 1
    rate_limit_per_minute: int = 60
    stream_max_seconds: int = 300
    stream_token_ttl_seconds: int = 60
    max_stream_sessions: int = 5
    job_retention_seconds: int = 3600
    timeline_segment_seconds: float = 3.0
    live_visual_window_seconds: float = 5.0
    live_text_min_interval_ms: int = 300

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        g = os.getenv
        s = cls(
            port=int(g("PORT", "8000")),
            api_token=g("API_TOKEN", ""),
            models_dir=g("MODELS_DIR", "/models"),
            tmp_dir=g("TMPDIR", "/tmp/emotions"),
            text_model_id=g("TEXT_MODEL_ID", cls.text_model_id),
            visual_repo_id=g("VISUAL_REPO_ID", cls.visual_repo_id),
            visual_backbone_file=g("VISUAL_BACKBONE_FILE", cls.visual_backbone_file),
            visual_lstm_file=g("VISUAL_LSTM_FILE", cls.visual_lstm_file),
            lstm_window=int(g("LSTM_WINDOW", "10")),
            allow_start_without_models=_bool("ALLOW_START_WITHOUT_MODELS", "true"),
            default_visual_weight=float(g("DEFAULT_VISUAL_WEIGHT", "0.5")),
            incongruence_threshold=float(g("INCONGRUENCE_THRESHOLD", "0.70")),
            max_image_bytes=int(float(g("MAX_IMAGE_MB", "5")) * 1024 * 1024),
            max_video_bytes=int(float(g("MAX_VIDEO_MB", "50")) * 1024 * 1024),
            max_video_seconds=float(g("MAX_VIDEO_SECONDS", "60")),
            max_text_chars=int(g("MAX_TEXT_CHARS", "5000")),
            max_frame_bytes=int(g("MAX_FRAME_BYTES", "2000000")),
            max_faces=int(g("MAX_FACES", "5")),
            max_concurrent_inferences=int(g("MAX_CONCURRENT_INFERENCES", "2")),
            max_video_jobs_queue=int(g("MAX_VIDEO_JOBS_QUEUE", "10")),
            max_video_workers=int(g("MAX_VIDEO_WORKERS", "1")),
            rate_limit_per_minute=int(g("RATE_LIMIT_PER_MINUTE", "60")),
            stream_max_seconds=int(g("STREAM_MAX_SECONDS", "300")),
            stream_token_ttl_seconds=int(g("STREAM_TOKEN_TTL_SECONDS", "60")),
            max_stream_sessions=int(g("MAX_STREAM_SESSIONS", "5")),
            job_retention_seconds=int(g("JOB_RETENTION_SECONDS", "3600")),
            timeline_segment_seconds=float(g("TIMELINE_SEGMENT_SECONDS", "3")),
            live_visual_window_seconds=float(g("LIVE_VISUAL_WINDOW_SECONDS", "5")),
            live_text_min_interval_ms=int(g("LIVE_TEXT_MIN_INTERVAL_MS", "300")),
        )
        return replace(s, **overrides) if overrides else s
