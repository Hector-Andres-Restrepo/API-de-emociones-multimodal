"""Multimodal en vivo: cámara + texto que se escribe, fusionados de forma continua.

REST:       POST /api/v1/emotions/live            (Bearer)  -> sessionId, websocketPath, streamToken efímero
WebSocket:  /api/v1/emotions/live/{sessionId}

Autenticación del WebSocket: el cliente envía como PRIMER mensaje de texto {"type":"auth","token":"<streamToken>"}
(token de un solo uso, nunca en la URL). Un token emitido para /emotions/streams no sirve aquí ni al revés.

Mensajes del cliente:  fotograma binario JPEG/PNG | {"type":"text","text":"...","version":n}
                       | {"type":"config","visualWeight":0..1} | {"type":"close"}
Mensajes del servidor: ready | update | dropped | error | closing    (ver docs/asyncapi.yaml)
"""
import asyncio
import json
import time
from typing import Optional

from fastapi import APIRouter, Depends, Request, WebSocket
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from ..deps import authenticate, require_text, require_visual, run_inference
from ..errors import ApiError, request_id_of
from ..services import media
from ..services.live import LiveAnalyzer

MODE = "MULTIMODAL_LIVE"
AUTH_TIMEOUT_S = 5

router = APIRouter(prefix="/api/v1", dependencies=[Depends(authenticate)])
ws_router = APIRouter(prefix="/api/v1")


class LiveIn(BaseModel):
    sampleFps: Optional[float] = None
    visualWeight: Optional[float] = None
    visualWindowSeconds: Optional[float] = None
    language: Optional[str] = "es"


@router.post("/emotions/live", status_code=201)
async def create_live(request: Request, body: LiveIn):
    st = request.app.state
    s = st.settings
    fps = 2.0 if body.sampleFps is None else float(body.sampleFps)
    if not 1.0 <= fps <= 5.0:
        raise ApiError(400, "VALIDATION_ERROR", "sampleFps debe estar entre 1 y 5.", {"field": "sampleFps"})
    weight = s.default_visual_weight if body.visualWeight is None else float(body.visualWeight)
    if not 0.0 <= weight <= 1.0:
        raise ApiError(400, "VALIDATION_ERROR", "visualWeight debe estar entre 0 y 1.", {"field": "visualWeight"})
    window = s.live_visual_window_seconds if body.visualWindowSeconds is None else float(body.visualWindowSeconds)
    if not 1.0 <= window <= 30.0:
        raise ApiError(400, "VALIDATION_ERROR", "visualWindowSeconds debe estar entre 1 y 30.", {"field": "visualWindowSeconds"})
    if (body.language or "es").lower() != "es":
        raise ApiError(422, "UNSUPPORTED_LANGUAGE", "Solo se admite español ('es') por ahora.", {"field": "language", "supported": ["es"]})
    text_eng, vis_eng = require_text(request), require_visual(request)
    sess, token = st.streams.create(fps, mode=MODE, params={"visual_weight": weight, "visual_window_s": window})
    return {
        "requestId": request_id_of(request),
        "mode": MODE,
        "sessionId": sess.session_id,
        "websocketPath": f"/api/v1/emotions/live/{sess.session_id}",
        "streamToken": token,
        "streamTokenExpiresInSeconds": s.stream_token_ttl_seconds,
        "authMessage": {"type": "auth", "token": "<streamToken>"},
        "sampleFps": fps,
        "visualWeight": weight,
        "visualWindowSeconds": window,
        "incongruenceThreshold": s.incongruence_threshold,
        "windowFrames": vis_eng.window,
        "maxFrameBytes": s.max_frame_bytes,
        "maxTextChars": s.max_text_chars,
        "textMinIntervalMs": s.live_text_min_interval_ms,
        "maxSessionSeconds": s.stream_max_seconds,
        "modelVersions": {"text": text_eng.version, "visual": vis_eng.version},
    }


async def _send_error(ws: WebSocket, session_id: str, e: ApiError) -> None:
    await ws.send_json({"type": "error", "sessionId": session_id,
                        "error": {"code": e.code, "message": e.message, "details": e.details}})


async def _send_simple_error(ws: WebSocket, session_id: str, code: str, message: str) -> None:
    await ws.send_json({"type": "error", "sessionId": session_id, "error": {"code": code, "message": message, "details": {}}})


@ws_router.websocket("/emotions/live/{session_id}")
async def live_ws(ws: WebSocket, session_id: str):
    st = ws.app.state
    s = st.settings
    await ws.accept()
    try:
        # 1) autenticación por primer mensaje (token efímero de un solo uso)
        try:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=AUTH_TIMEOUT_S)
            msg = json.loads(raw)
            token = msg.get("token") if isinstance(msg, dict) and msg.get("type") == "auth" else None
        except WebSocketDisconnect:
            return
        except Exception:
            token = None
        sess = st.streams.authenticate(session_id, token, mode=MODE) if token else None
        if sess is None:
            await _send_simple_error(ws, session_id, "UNAUTHORIZED", "Token de sesión ausente, inválido, expirado, ya usado o de otro tipo de sesión.")
            await ws.close(code=1008)
            return
        if not (st.visual.ready and st.text.ready):
            await _send_simple_error(ws, session_id, "MODEL_UNAVAILABLE", "Los modelos no están cargados.")
            await ws.close(code=1011)
            return

        analyzer = LiveAnalyzer(st.text, st.visual, s, sample_fps=sess.sample_fps,
                                visual_weight=sess.params["visual_weight"], visual_window_s=sess.params["visual_window_s"])
        versions = {"text": st.text.version, "visual": st.visual.version}
        await ws.send_json({
            "type": "ready", "sessionId": session_id, "mode": MODE, "sampleFps": sess.sample_fps,
            "visualWeight": analyzer.visual_weight, "visualWindowSeconds": analyzer.window_s,
            "incongruenceThreshold": s.incongruence_threshold, "windowFrames": st.visual.window,
            "maxFrameBytes": s.max_frame_bytes, "maxTextChars": s.max_text_chars,
            "textMinIntervalMs": s.live_text_min_interval_ms, "modelVersions": versions,
        })

        async def send_update(fn, *args):
            upd = await run_in_threadpool(run_inference, st, fn, *args)
            upd.update(sessionId=session_id, modelVersions=versions)
            await ws.send_json(upd)

        async def closing(reason: str, code: int = 1000):
            await ws.send_json({"type": "closing", "sessionId": session_id, "reason": reason, "summary": analyzer.summary()})
            await ws.close(code=code)

        # 2) bucle de eventos: fotogramas binarios y mensajes JSON
        while True:
            remaining = s.stream_max_seconds - (time.monotonic() - analyzer.started)
            if remaining <= 0:
                await closing("SESSION_TIMEOUT")
                return
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
            except asyncio.TimeoutError:
                continue
            if msg["type"] == "websocket.disconnect":
                return

            if msg.get("text") is not None:  # ---- mensaje de control o de texto
                raw = msg["text"]
                if len(raw) > s.max_text_chars * 4 + 2048:
                    await _send_simple_error(ws, session_id, "VALIDATION_ERROR", "Mensaje demasiado grande.")
                    continue
                try:
                    ctrl = json.loads(raw)
                except ValueError:
                    ctrl = None
                if not isinstance(ctrl, dict):
                    await _send_simple_error(ws, session_id, "VALIDATION_ERROR", "Se esperaba un objeto JSON con el campo 'type'.")
                    continue
                kind = ctrl.get("type")
                try:
                    if kind == "close":
                        await closing("CLIENT_CLOSE")
                        return
                    elif kind == "text":
                        wait_ms = analyzer.check_text_rate()
                        if wait_ms:
                            await ws.send_json({"type": "dropped", "kind": "TEXT", "sessionId": session_id, "retryAfterMs": wait_ms,
                                                "warnings": ["TEXT_DROPPED_RATE_LIMIT: reenvíe el último texto tras retryAfterMs."]})
                            continue
                        version = ctrl.get("version")
                        await send_update(analyzer.on_text, ctrl.get("text"), version if isinstance(version, int) and not isinstance(version, bool) else None)
                    elif kind == "config":
                        await send_update(analyzer.set_visual_weight, ctrl.get("visualWeight"))
                    else:
                        await _send_simple_error(ws, session_id, "VALIDATION_ERROR", "Tipo de mensaje no admitido. Use 'text', 'config' o 'close'.")
                except ApiError as e:
                    await _send_error(ws, session_id, e)
                continue

            # ---- fotograma binario
            data: bytes = msg.get("bytes") or b""
            if len(data) > s.max_frame_bytes:
                await _send_simple_error(ws, session_id, "PAYLOAD_TOO_LARGE", f"Fotograma mayor a {s.max_frame_bytes} bytes.")
                await ws.close(code=1009)
                return
            if not analyzer.check_frame_rate():
                await ws.send_json({"type": "dropped", "kind": "FRAME", "sessionId": session_id,
                                    "warnings": ["FRAME_DROPPED_RATE_LIMIT: envíe a lo sumo sampleFps cuadros por segundo."]})
                continue
            try:
                img = media.decode_image(data, "frame", s.max_image_pixels)
                del data
                await send_update(analyzer.on_frame, img)
            except ApiError as e:
                await _send_error(ws, session_id, e)
    except WebSocketDisconnect:
        pass
    finally:
        st.streams.remove(session_id)
        try:
            await ws.close()
        except Exception:
            pass
