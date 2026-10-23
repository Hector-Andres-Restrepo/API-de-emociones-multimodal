"""Pruebas del WebSocket multimodal en vivo (MOCKS SOLO EN PRUEBAS)."""
import logging
import time

import pytest
from starlette.websockets import WebSocketDisconnect

from tests.conftest import AUTH, jpeg_bytes
from tests.fakes import FakeVisualEngine, KeywordTextEngine, onehot


def open_live(c, **body):
    r = c.post("/api/v1/emotions/live", json=body, headers=AUTH)
    assert r.status_code == 201, r.text
    return r.json()


def auth_msg(s):
    return {"type": "auth", "token": s["streamToken"]}


def start(ws, s):
    ws.send_json(auth_msg(s))
    m = ws.receive_json()
    assert m["type"] == "ready"
    return m


def test_create_live_contract(client):
    s = open_live(client, sampleFps=2, visualWeight=0.4, visualWindowSeconds=3)
    assert s["mode"] == "MULTIMODAL_LIVE" and s["websocketPath"] == f"/api/v1/emotions/live/{s['sessionId']}"
    assert s["visualWeight"] == 0.4 and s["visualWindowSeconds"] == 3 and s["incongruenceThreshold"] == 0.70
    assert s["windowFrames"] == 4 and s["maxTextChars"] and s["textMinIntervalMs"] == 300
    assert s["modelVersions"]["text"] and s["modelVersions"]["visual"]
    assert "token" not in s["websocketPath"] and s["authMessage"]["type"] == "auth"


def test_create_live_defaults(client):
    s = open_live(client)
    assert s["sampleFps"] == 2.0 and s["visualWeight"] == 0.5 and s["visualWindowSeconds"] == 5.0


@pytest.mark.parametrize("body", [{"sampleFps": 0}, {"sampleFps": 6}, {"visualWeight": 1.2}, {"visualWeight": -1},
                                  {"visualWindowSeconds": 0.5}, {"visualWindowSeconds": 31}])
def test_create_live_invalid_params(client, body):
    r = client.post("/api/v1/emotions/live", json=body, headers=AUTH)
    assert r.status_code == 400 and r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_create_live_unsupported_language(client):
    r = client.post("/api/v1/emotions/live", json={"language": "en"}, headers=AUTH)
    assert r.status_code == 422 and r.json()["error"]["code"] == "UNSUPPORTED_LANGUAGE"


def test_create_live_requires_bearer(client):
    assert client.post("/api/v1/emotions/live", json={}).status_code == 401


def test_create_live_503_when_a_model_is_missing(make_client):
    text = KeywordTextEngine()
    text.ready = False
    assert make_client(text=text).post("/api/v1/emotions/live", json={}, headers=AUTH).status_code == 503
    vis = FakeVisualEngine()
    vis.ready = False
    assert make_client(visual=vis).post("/api/v1/emotions/live", json={}, headers=AUTH).status_code == 503


def test_live_shares_session_limit_with_streams(make_client):
    c = make_client(max_stream_sessions=2)
    open_live(c)
    assert c.post("/api/v1/emotions/streams", json={}, headers=AUTH).status_code == 201
    assert c.post("/api/v1/emotions/live", json={}, headers=AUTH).status_code == 429


def test_ws_ready_message(client):
    s = open_live(client, visualWeight=0.3)
    with client.websocket_connect(s["websocketPath"]) as ws:
        m = start(ws, s)
        assert m["mode"] == "MULTIMODAL_LIVE" and m["visualWeight"] == 0.3 and m["windowFrames"] == 4
        assert m["textMinIntervalMs"] == 300 and m["modelVersions"]["text"]


def test_ws_roundtrip_text_and_frames_with_incongruence(make_client):
    c = make_client(text=KeywordTextEngine(), visual=FakeVisualEngine(temporal=onehot("HAPPINESS", 0.9)))
    s = open_live(c)
    with c.websocket_connect(s["websocketPath"]) as ws:
        start(ws, s)
        ws.send_bytes(jpeg_bytes())
        m = ws.receive_json()
        assert m["type"] == "update" and m["trigger"] == "FRAME" and m["fusionStatus"] == "WAITING_FOR_TEXT"
        assert m["sessionId"] == s["sessionId"] and m["modelVersions"]["visual"]
        ws.send_json({"type": "text", "text": "Estoy muy triste", "version": 1})
        m = ws.receive_json()
        assert m["trigger"] == "TEXT" and m["fusionStatus"] == "OK" and m["text"]["textVersion"] == 1
        assert m["fusion"]["incongruence"] is True and m["fusion"]["emotion"] in ("HAPPINESS", "SADNESS")
        time.sleep(0.35)
        ws.send_json({"type": "text", "text": "Estoy muy feliz", "version": 2})
        m = ws.receive_json()
        assert m["fusion"]["incongruence"] is False and m["fusion"]["emotion"] == "HAPPINESS"
        ws.send_json({"type": "close"})
        z = ws.receive_json()
        assert z["type"] == "closing" and z["reason"] == "CLIENT_CLOSE"
        assert z["summary"] == {"framesProcessed": 1, "textsAnalyzed": 2, "fusionsComputed": 2, "incongruentFusions": 1}


def test_ws_config_changes_visual_weight(make_client):
    c = make_client(text=KeywordTextEngine(), visual=FakeVisualEngine(temporal=onehot("HAPPINESS", 0.85)))
    s = open_live(c)
    with c.websocket_connect(s["websocketPath"]) as ws:
        start(ws, s)
        ws.send_bytes(jpeg_bytes()); ws.receive_json()
        ws.send_json({"type": "text", "text": "Estoy triste"}); ws.receive_json()
        ws.send_json({"type": "config", "visualWeight": 0})
        m = ws.receive_json()
        assert m["trigger"] == "CONFIG" and m["fusion"]["emotion"] == "SADNESS" and m["fusion"]["visualWeight"] == 0.0
        ws.send_json({"type": "config", "visualWeight": 7})
        e = ws.receive_json()
        assert e["type"] == "error" and e["error"]["code"] == "VALIDATION_ERROR"
        ws.send_json({"type": "config", "visualWeight": 1})
        assert ws.receive_json()["fusion"]["emotion"] == "HAPPINESS"  # la sesión continúa


def test_ws_text_too_soon_is_dropped_with_retry(make_client):
    c = make_client(live_text_min_interval_ms=1500)
    s = open_live(c)
    with c.websocket_connect(s["websocketPath"]) as ws:
        start(ws, s)
        ws.send_json({"type": "text", "text": "hola"}); assert ws.receive_json()["type"] == "update"
        ws.send_json({"type": "text", "text": "hola otra vez"})
        m = ws.receive_json()
        assert m["type"] == "dropped" and m["kind"] == "TEXT" and 0 < m["retryAfterMs"] <= 1500


def test_ws_frames_too_fast_are_dropped(client):
    s = open_live(client, sampleFps=1)
    with client.websocket_connect(s["websocketPath"]) as ws:
        start(ws, s)
        ws.send_bytes(jpeg_bytes()); assert ws.receive_json()["type"] == "update"
        ws.send_bytes(jpeg_bytes())
        m = ws.receive_json()
        assert m["type"] == "dropped" and m["kind"] == "FRAME"


def test_ws_validation_errors_keep_session(client):
    s = open_live(client)
    with client.websocket_connect(s["websocketPath"]) as ws:
        start(ws, s)
        ws.send_text("no es json")
        assert ws.receive_json()["error"]["code"] == "VALIDATION_ERROR"
        ws.send_json({"type": "otra-cosa"})
        assert ws.receive_json()["error"]["code"] == "VALIDATION_ERROR"
        ws.send_bytes(b"no soy una imagen")
        assert ws.receive_json()["error"]["code"] == "INVALID_FILE_FORMAT"
        time.sleep(0.35)
        ws.send_json({"type": "text", "text": 123})
        assert ws.receive_json()["error"]["code"] == "VALIDATION_ERROR"
        time.sleep(0.35)
        ws.send_json({"type": "text", "text": "hola"})
        assert ws.receive_json()["type"] == "update"


def test_ws_text_over_limit(make_client):
    c = make_client(max_text_chars=20)
    s = open_live(c)
    with c.websocket_connect(s["websocketPath"]) as ws:
        start(ws, s)
        ws.send_json({"type": "text", "text": "x" * 21})
        assert ws.receive_json()["error"]["code"] == "VALIDATION_ERROR"


def test_ws_frame_too_large_closes(make_client):
    c = make_client(max_frame_bytes=500)
    s = open_live(c)
    with c.websocket_connect(s["websocketPath"]) as ws:
        start(ws, s)
        ws.send_bytes(jpeg_bytes(200, 200))
        assert ws.receive_json()["error"]["code"] == "PAYLOAD_TOO_LARGE"
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
        assert e.value.code == 1009


def test_ws_multiple_faces_block_fusion(make_client):
    c = make_client(visual=FakeVisualEngine(faces=2))
    s = open_live(c)
    with c.websocket_connect(s["websocketPath"]) as ws:
        start(ws, s)
        ws.send_json({"type": "text", "text": "Estoy triste"}); ws.receive_json()
        ws.send_bytes(jpeg_bytes())
        m = ws.receive_json()
        assert m["facesDetected"] == 2 and m["fusion"] is None
        assert m["fusionStatus"] == "FACE_COUNT_NOT_SUPPORTED_FOR_FUSION"


# ------------------------------------------------------------------ autenticación
def test_ws_bad_token_and_single_use(client):
    s = open_live(client)
    with client.websocket_connect(s["websocketPath"]) as ws:
        ws.send_json({"type": "auth", "token": "falso"})
        assert ws.receive_json()["error"]["code"] == "UNAUTHORIZED"
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
        assert e.value.code == 1008
    s2 = open_live(client)
    with client.websocket_connect(s2["websocketPath"]) as ws:
        start(ws, s2)
    with client.websocket_connect(s2["websocketPath"]) as ws:
        ws.send_json(auth_msg(s2))
        assert ws.receive_json()["error"]["code"] == "UNAUTHORIZED"


def test_ws_unknown_session_and_missing_auth_message(client):
    with client.websocket_connect("/api/v1/emotions/live/inexistente") as ws:
        ws.send_json({"type": "auth", "token": "x"})
        assert ws.receive_json()["error"]["code"] == "UNAUTHORIZED"
    s = open_live(client)
    with client.websocket_connect(s["websocketPath"]) as ws:
        ws.send_json({"type": "text", "text": "hola"})  # sin autenticar
        assert ws.receive_json()["error"]["code"] == "UNAUTHORIZED"


def test_tokens_are_not_interchangeable_between_session_types(client):
    live = open_live(client)
    st = client.post("/api/v1/emotions/streams", json={}, headers=AUTH).json()
    with client.websocket_connect(live["websocketPath"].replace("/live/", "/streams/")) as ws:
        ws.send_json(auth_msg(live))
        assert ws.receive_json()["error"]["code"] == "UNAUTHORIZED"
    with client.websocket_connect(st["websocketPath"].replace("/streams/", "/live/")) as ws:
        ws.send_json(auth_msg(st))
        assert ws.receive_json()["error"]["code"] == "UNAUTHORIZED"


# ------------------------------------------------------------------ cierre y privacidad
def test_ws_session_timeout(make_client):
    c = make_client(stream_max_seconds=1)
    s = open_live(c)
    with c.websocket_connect(s["websocketPath"]) as ws:
        start(ws, s)
        m = ws.receive_json()
        assert m["type"] == "closing" and m["reason"] == "SESSION_TIMEOUT" and "summary" in m


def test_ws_disconnect_frees_session(client):
    s = open_live(client)
    with client.websocket_connect(s["websocketPath"]) as ws:
        start(ws, s)
    time.sleep(0.3)
    assert s["sessionId"] not in client.app.state.streams._sessions


def test_text_is_not_logged(client, caplog):
    secret = "confidencial-texto-privado-424242"
    s = open_live(client)
    with caplog.at_level(logging.DEBUG):
        with client.websocket_connect(s["websocketPath"]) as ws:
            start(ws, s)
            ws.send_json({"type": "text", "text": secret})
            m = ws.receive_json()
            assert secret not in str(m)
    assert secret not in caplog.text
