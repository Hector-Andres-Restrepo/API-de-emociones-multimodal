"""Pruebas del WebSocket de cámara (MOCKS SOLO EN PRUEBAS)."""
import time

import pytest
from starlette.websockets import WebSocketDisconnect

from tests.conftest import AUTH, jpeg_bytes
from tests.fakes import FakeVisualEngine, onehot


def open_session(c, fps=5):
    r = c.post("/api/v1/emotions/streams", json={"sampleFps": fps}, headers=AUTH)
    assert r.status_code == 201
    return r.json()


def auth_msg(s):
    return {"type": "auth", "token": s["streamToken"]}


def test_create_session_contract(client):
    s = open_session(client, 2)
    assert s["sessionId"] and s["websocketPath"].endswith(s["sessionId"]) and s["modelVersions"]["visual"]
    assert s["windowFrames"] == 4 and s["streamToken"] and "token" not in s["websocketPath"]


@pytest.mark.parametrize("fps", [0, 6, -1])
def test_create_session_invalid_fps(client, fps):
    assert client.post("/api/v1/emotions/streams", json={"sampleFps": fps}, headers=AUTH).status_code == 400


def test_stream_requires_bearer_to_create(client):
    assert client.post("/api/v1/emotions/streams", json={}).status_code == 401


def test_ws_frame_roundtrip_temporal_window(make_client):
    vis = FakeVisualEngine(temporal=onehot("SADNESS", 0.7))
    c = make_client(visual=vis)
    s = open_session(c)
    with c.websocket_connect(s["websocketPath"]) as ws:
        ws.send_json(auth_msg(s))
        assert ws.receive_json()["type"] == "ready"
        for _ in range(3):
            ws.send_bytes(jpeg_bytes())
            m = ws.receive_json()
            assert m["sessionId"] == s["sessionId"] and m["facesDetected"] == 1
            assert m["faces"][0]["emotion"] == "SADNESS" and m["faces"][0]["mode"] == "LSTM"
            assert m["temporal"]["windowFrames"] == 4 and m["modelVersions"]["visual"] and "inferenceMs" in m
            time.sleep(0.25)
        ws.send_json({"type": "close"})
        assert ws.receive_json()["type"] == "closing"
    assert all(sh == (4, 512) for sh in vis.window_shapes) and len(vis.window_shapes) == 3


def test_ws_no_face(make_client):
    c = make_client(visual=FakeVisualEngine(faces=0))
    s = open_session(c)
    with c.websocket_connect(s["websocketPath"]) as ws:
        ws.send_json(auth_msg(s)); ws.receive_json()
        ws.send_bytes(jpeg_bytes())
        m = ws.receive_json()
        assert m["facesDetected"] == 0 and m["faces"] == [] and "NO_FACE_DETECTED" in m["warnings"]


def test_ws_rejects_bad_token_and_token_is_single_use(client):
    s = open_session(client)
    with client.websocket_connect(s["websocketPath"]) as ws:
        ws.send_json({"type": "auth", "token": "falso"})
        assert ws.receive_json()["error"]["code"] == "UNAUTHORIZED"
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()
    s2 = open_session(client)
    with client.websocket_connect(s2["websocketPath"]) as ws:
        ws.send_json(auth_msg(s2)); assert ws.receive_json()["type"] == "ready"
    with client.websocket_connect(s2["websocketPath"]) as ws:  # reutilizar el token falla
        ws.send_json(auth_msg(s2))
        assert ws.receive_json()["error"]["code"] == "UNAUTHORIZED"


def test_ws_rejects_unknown_session(client):
    with client.websocket_connect("/api/v1/emotions/streams/session-inexistente") as ws:
        ws.send_json({"type": "auth", "token": "x"})
        assert ws.receive_json()["error"]["code"] == "UNAUTHORIZED"


def test_ws_frame_too_large_closes(make_client):
    c = make_client(max_frame_bytes=500)
    s = open_session(c)
    with c.websocket_connect(s["websocketPath"]) as ws:
        ws.send_json(auth_msg(s)); ws.receive_json()
        ws.send_bytes(jpeg_bytes(200, 200))
        assert ws.receive_json()["error"]["code"] == "PAYLOAD_TOO_LARGE"
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
        assert e.value.code == 1009


def test_ws_invalid_frame_keeps_session(client):
    s = open_session(client)
    with client.websocket_connect(s["websocketPath"]) as ws:
        ws.send_json(auth_msg(s)); ws.receive_json()
        ws.send_bytes(b"no soy una imagen")
        assert ws.receive_json()["error"]["code"] == "INVALID_FILE_FORMAT"
        time.sleep(0.25)
        ws.send_bytes(jpeg_bytes())
        assert ws.receive_json()["facesDetected"] == 1


def test_ws_rate_limit_drops_fast_frames(client):
    s = open_session(client, fps=1)
    with client.websocket_connect(s["websocketPath"]) as ws:
        ws.send_json(auth_msg(s)); ws.receive_json()
        ws.send_bytes(jpeg_bytes()); assert ws.receive_json()["facesDetected"] == 1
        ws.send_bytes(jpeg_bytes())
        m = ws.receive_json()
        assert m.get("dropped") is True and "FRAME_DROPPED_RATE_LIMIT" in m["warnings"][0]


def test_ws_session_timeout(make_client):
    c = make_client(stream_max_seconds=1)
    s = open_session(c)
    with c.websocket_connect(s["websocketPath"]) as ws:
        ws.send_json(auth_msg(s)); ws.receive_json()
        m = ws.receive_json()
        assert m["type"] == "closing" and m["reason"] == "SESSION_TIMEOUT"


def test_ws_safe_disconnect_frees_session(client):
    s = open_session(client)
    with client.websocket_connect(s["websocketPath"]) as ws:
        ws.send_json(auth_msg(s)); ws.receive_json()
    time.sleep(0.3)
    assert s["sessionId"] not in client.app.state.streams._sessions


def test_max_stream_sessions(make_client):
    c = make_client(max_stream_sessions=2)
    open_session(c); open_session(c)
    assert c.post("/api/v1/emotions/streams", json={}, headers=AUTH).status_code == 429
