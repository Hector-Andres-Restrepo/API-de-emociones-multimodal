"""Cámara en vivo: POST /emotions/streams (REST, Bearer) + WebSocket /emotions/streams/{sessionId}.

Autenticación del WebSocket: el cliente recibe un `streamToken` efímero (un solo uso, TTL corto)
al crear la sesión y lo envía en el PRIMER mensaje de texto {"type":"auth","token":"..."};
nunca viaja en la URL.
"""
import asyncio
import json
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, Request, WebSocket
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from ..deps import DISCLAIMER_FACE, authenticate, require_visual, run_inference
from ..errors import ApiError, request_id_of
from ..labels import top
from ..services import media

router = APIRouter(prefix="/api/v1", dependencies=[Depends(authenticate)])
ws_router = APIRouter(prefix="/api/v1")

AUTH_TIMEOUT_S = 5


class StreamIn(BaseModel):
    sampleFps: Optional[float] = 2


@router.post("/emotions/streams", status_code=201)
async def create_stream(request: Request, body: StreamIn):
    st = request.app.state
    fps = 2.0 if body.sampleFps is None else float(body.sampleFps)
    if not 1.0 <= fps <= 5.0:
        raise ApiError(400, "VALIDATION_ERROR", "sampleFps debe estar entre 1 y 5.", {"field": "sampleFps"})
    eng = require_visual(request)
    sess, token = st.streams.create(fps)
    return {
        "requestId": request_id_of(request),
        "sessionId": sess.session_id,
        "websocketPath": f"/api/v1/emotions/streams/{sess.session_id}",
        "streamToken": token,
        "streamTokenExpiresInSeconds": st.settings.stream_token_ttl_seconds,
        "sampleFps": fps,
        "windowFrames": eng.window,
        "maxFrameBytes": st.settings.max_frame_bytes,
        "maxSessionSeconds": st.settings.stream_max_seconds,
        "authMessage": {"type": "auth", "token": "<streamToken>"},
        "modelVersions": {"visual": eng.version},
    }


async def _send_error(ws: WebSocket, session_id: str, code: str, message: str) -> None:
    await ws.send_json({"sessionId": session_id, "error": {"code": code, "message": message}})


@ws_router.websocket("/emotions/streams/{session_id}")
async def stream_ws(ws: WebSocket, session_id: str):
    st = ws.app.state
    s = st.settings
    await ws.accept()
    sess = None
    try:
        # 1) autenticación por primer mensaje
        try:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=AUTH_TIMEOUT_S)
            msg = json.loads(raw)
            token = msg.get("token") if isinstance(msg, dict) and msg.get("type") == "auth" else None
        except WebSocketDisconnect:
            return
        except Exception:
            token = None
        sess = st.streams.authenticate(session_id, token) if token else None
        if sess is None:
            await _send_error(ws, session_id, "UNAUTHORIZED", "Token de sesión ausente, inválido, expirado o ya usado.")
            await ws.close(code=1008)
            return
        if not st.visual.ready:
            await _send_error(ws, session_id, "MODEL_UNAVAILABLE", "El modelo visual no está cargado.")
            await ws.close(code=1011)
            return
        await ws.send_json({"sessionId": session_id, "type": "ready", "sampleFps": sess.sample_fps,
                            "windowFrames": st.visual.window, "modelVersions": {"visual": st.visual.version}})

        # 2) bucle de fotogramas
        started = time.monotonic()
        min_interval = 0.8 / sess.sample_fps  # tolerancia del 20 %
        while True:
            remaining = s.stream_max_seconds - (time.monotonic() - started)
            if remaining <= 0:
                await ws.send_json({"sessionId": session_id, "type": "closing", "reason": "SESSION_TIMEOUT"})
                await ws.close(code=1000)
                return
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
            except asyncio.TimeoutError:
                continue
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("text") is not None:  # mensaje de control
                try:
                    ctrl = json.loads(msg["text"])
                except ValueError:
                    ctrl = {}
                if isinstance(ctrl, dict) and ctrl.get("type") == "close":
                    await ws.send_json({"sessionId": session_id, "type": "closing", "reason": "CLIENT_CLOSE",
                                        "framesProcessed": sess.frames_processed})
                    await ws.close(code=1000)
                    return
                await _send_error(ws, session_id, "VALIDATION_ERROR", "Se esperaban fotogramas binarios JPEG/PNG.")
                continue
            data: bytes = msg.get("bytes") or b""
            ts_ms = int((time.monotonic() - started) * 1000)
            if len(data) > s.max_frame_bytes:
                await _send_error(ws, session_id, "PAYLOAD_TOO_LARGE", f"Fotograma mayor a {s.max_frame_bytes} bytes.")
                await ws.close(code=1009)
                return
            now = time.monotonic()
            if sess.last_frame_at and now - sess.last_frame_at < min_interval:
                await ws.send_json({"sessionId": session_id, "timestampMs": ts_ms, "dropped": True,
                                    "warnings": ["FRAME_DROPPED_RATE_LIMIT: envíe a lo sumo sampleFps cuadros por segundo."]})
                continue
            sess.last_frame_at = now
            try:
                img = media.decode_image(data, "frame", s.max_image_pixels)
            except ApiError as e:
                await _send_error(ws, session_id, e.code, e.message)
                continue
            del data
            t0 = time.perf_counter()

            def work():
                faces = []
                for i, box in enumerate(st.visual.detect_faces(img)[: s.max_faces], start=1):
                    probs, feat = st.visual.embed(img, box)
                    mode = "STATIC_FRAME"
                    if i == 1:  # rostro principal: modelo temporal LSTM con ventana por sesión
                        if not sess.window:
                            sess.window.extend([feat] * st.visual.window)
                        else:
                            sess.window.append(feat)
                        probs, mode = st.visual.temporal(np.stack(sess.window)), "LSTM"
                    emo, conf = top(probs)
                    faces.append({"faceId": f"face-{i}", "emotion": emo, "confidence": round(conf, 4), "mode": mode})
                return faces

            try:
                faces = await run_in_threadpool(run_inference, st, work)
            except ApiError as e:
                await _send_error(ws, session_id, e.code, e.message)
                continue
            sess.frames_processed += 1
            warnings = [] if faces else ["NO_FACE_DETECTED"]
            await ws.send_json({
                "sessionId": session_id, "timestampMs": ts_ms, "facesDetected": len(faces), "faces": faces,
                "temporal": {"windowFrames": st.visual.window, "windowFilled": len(sess.window)},
                "inferenceMs": int((time.perf_counter() - t0) * 1000),
                "modelVersions": {"visual": st.visual.version},
                "warnings": [*warnings, DISCLAIMER_FACE] if faces else warnings,
            })
    except WebSocketDisconnect:
        pass
    finally:
        st.streams.remove(session_id)
        try:
            await ws.close()
        except Exception:
            pass
