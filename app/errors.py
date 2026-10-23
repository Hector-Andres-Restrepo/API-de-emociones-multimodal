"""Contrato de errores común (sección 6)."""
import logging
import uuid
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("emotions.errors")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: Optional[dict] = None):
        self.status, self.code, self.message, self.details = status, code, message, details or {}


def request_id_of(request: Request) -> str:
    return getattr(request.state, "request_id", None) or str(uuid.uuid4())


def error_body(request_id: str, code: str, message: str, details: Optional[dict] = None) -> dict:
    return {"requestId": request_id, "error": {"code": code, "message": message, "details": details or {}}}


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return JSONResponse(error_body(request_id_of(request), exc.code, exc.message, exc.details), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        fields = sorted({".".join(str(p) for p in e.get("loc", ())[1:]) or "body" for e in exc.errors()})
        return JSONResponse(
            error_body(request_id_of(request), "VALIDATION_ERROR", "Parámetros faltantes o inválidos.", {"fields": fields}),
            status_code=400,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        mapping = {401: "UNAUTHORIZED", 404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED", 413: "PAYLOAD_TOO_LARGE"}
        code = mapping.get(exc.status_code, "VALIDATION_ERROR" if exc.status_code < 500 else "INFERENCE_ERROR")
        return JSONResponse(error_body(request_id_of(request), code, str(exc.detail)), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        # No se registra el contenido de la solicitud (privacidad): solo el tipo de excepción.
        log.error("Error no controlado: %s", type(exc).__name__)
        return JSONResponse(error_body(request_id_of(request), "INFERENCE_ERROR", "Error interno de inferencia."), status_code=500)
