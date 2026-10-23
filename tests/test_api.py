"""Pruebas de contrato REST con motores simulados (MOCKS SOLO EN PRUEBAS)."""
import os
import uuid

import pytest

from app.labels import CANONICAL
from tests.conftest import AUTH, jpeg_bytes, make_mp4, png_bytes, wait_job
from tests.fakes import FakeTextEngine, FakeVisualEngine, onehot


def is_uuid(v):
    return str(uuid.UUID(v)) == v


# ---------------------------------------------------------------- salud y auth
def test_health(client, make_client):
    assert client.get("/health/live").json() == {"status": "alive"}
    r = client.get("/health/ready")
    assert r.status_code == 200 and r.json()["models"] == {"text": True, "visual": True}
    f = FakeVisualEngine()
    f.ready = False
    c2 = make_client(visual=f)
    assert c2.get("/health/ready").status_code == 503


def test_demo_page_is_public_static_html(client):
    r = client.get("/demo")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Demo" in r.text and "no determina la emoción interna" in r.text
    assert "cambia-este-token" not in r.text  # la página no contiene secretos


def test_auth_required_everywhere(client):
    for method, url in [("post", "/api/v1/emotions/text"), ("post", "/api/v1/emotions/image"),
                        ("post", "/api/v1/emotions/video"), ("post", "/api/v1/emotions/analyze"),
                        ("get", "/api/v1/emotions/video/vid-x"), ("post", "/api/v1/emotions/streams")]:
        r = getattr(client, method)(url)
        assert r.status_code == 401, url
        assert r.json()["error"]["code"] == "UNAUTHORIZED" and is_uuid(r.json()["requestId"])
    r = client.post("/api/v1/emotions/text", json={"text": "hola"}, headers={"Authorization": "Bearer malo"})
    assert r.status_code == 401


def test_rate_limit(make_client):
    c = make_client(rate_limit_per_minute=3)
    codes = [c.post("/api/v1/emotions/text", json={"text": "hola"}, headers=AUTH).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]


# ---------------------------------------------------------------- texto
def test_text_ok(client):
    r = client.post("/api/v1/emotions/text", json={"text": "Hoy me siento muy feliz", "language": "es"}, headers=AUTH)
    b = r.json()
    assert r.status_code == 200 and is_uuid(b["requestId"])
    assert b["modality"] == "TEXT" and b["emotion"] == "HAPPINESS"
    assert set(b["probabilities"]) == set(CANONICAL) and sum(b["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert b["modelVersions"]["text"] and b["warnings"]


@pytest.mark.parametrize("body", [{}, {"text": ""}, {"text": "   "}, {"text": "x" * 5001}])
def test_text_validation(client, body):
    r = client.post("/api/v1/emotions/text", json=body, headers=AUTH)
    assert r.status_code == 400 and r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_text_max_length_boundary(client):
    assert client.post("/api/v1/emotions/text", json={"text": "a" * 5000}, headers=AUTH).status_code == 200


def test_text_unsupported_language(client):
    r = client.post("/api/v1/emotions/text", json={"text": "hello", "language": "en"}, headers=AUTH)
    assert r.status_code == 422 and r.json()["error"]["code"] == "UNSUPPORTED_LANGUAGE"


def test_model_unavailable_503(make_client):
    t = FakeTextEngine()
    t.ready = False
    r = make_client(text=t).post("/api/v1/emotions/text", json={"text": "hola"}, headers=AUTH)
    assert r.status_code == 503 and r.json()["error"]["code"] == "MODEL_UNAVAILABLE"


# ---------------------------------------------------------------- imagen
def test_image_one_face(client):
    r = client.post("/api/v1/emotions/image", files={"image": ("a.jpg", jpeg_bytes(), "image/jpeg")}, headers=AUTH)
    b = r.json()
    assert r.status_code == 200 and b["modality"] == "IMAGE" and b["facesDetected"] == 1
    assert b["faces"][0]["faceId"] == "face-1" and b["faces"][0]["emotion"] == "HAPPINESS"
    assert b["modelVersions"]["visual"] and any("no determinan" in w for w in b["warnings"])


def test_image_png_ok(client):
    assert client.post("/api/v1/emotions/image", files={"image": ("a.png", png_bytes(), "image/png")}, headers=AUTH).status_code == 200


def test_image_no_face(make_client):
    c = make_client(visual=FakeVisualEngine(faces=0))
    b = c.post("/api/v1/emotions/image", files={"image": ("a.jpg", jpeg_bytes(), "image/jpeg")}, headers=AUTH).json()
    assert b["facesDetected"] == 0 and b["faces"] == [] and "NO_FACE_DETECTED" in b["warnings"]


def test_image_multiple_faces_reported_separately(make_client):
    c = make_client(visual=FakeVisualEngine(faces=3))
    b = c.post("/api/v1/emotions/image", files={"image": ("a.jpg", jpeg_bytes(), "image/jpeg")}, headers=AUTH).json()
    assert b["facesDetected"] == 3 and [f["faceId"] for f in b["faces"]] == ["face-1", "face-2", "face-3"]


def test_image_invalid_format_415(client):
    r = client.post("/api/v1/emotions/image", files={"image": ("a.gif", b"GIF89a....", "image/gif")}, headers=AUTH)
    assert r.status_code == 415 and r.json()["error"]["code"] == "INVALID_FILE_FORMAT"
    assert r.json()["error"]["details"]["field"] == "image"


def test_image_too_large_413(make_client):
    c = make_client(max_image_bytes=1000)
    r = c.post("/api/v1/emotions/image", files={"image": ("a.jpg", jpeg_bytes(200, 200), "image/jpeg")}, headers=AUTH)
    assert r.status_code == 413 and r.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_image_missing_field_400(client):
    r = client.post("/api/v1/emotions/image", files={"otro": ("a.jpg", jpeg_bytes(), "image/jpeg")}, headers=AUTH)
    assert r.status_code == 400 and r.json()["error"]["code"] == "VALIDATION_ERROR"


# ---------------------------------------------------------------- video asíncrono
def test_video_lifecycle_uses_temporal_model(make_client, mp4, tmp_path):
    vis = FakeVisualEngine(temporal=onehot("SURPRISE", 0.8))
    c = make_client(visual=vis)
    r = c.post("/api/v1/emotions/video", files={"video": ("v.mp4", mp4, "video/mp4")}, data={"sampleFps": "2"}, headers=AUTH)
    assert r.status_code == 202
    b = r.json()
    assert b["analysisId"].startswith("vid-") and b["status"] in ("PENDING", "PROCESSING", "COMPLETED")
    assert b["resultUrl"] == f"/api/v1/emotions/video/{b['analysisId']}" and is_uuid(b["requestId"])
    res = wait_job(c, b["resultUrl"])
    j = res.json()
    assert j["status"] == "COMPLETED" and j["dominantExpression"] == "SURPRISE"
    assert j["timeline"] and j["timeline"][0]["startMs"] == 0
    assert j["temporalModel"]["type"] == "LSTM" and j["temporalModel"]["windowFrames"] == vis.window
    # el LSTM recibió ventanas (W, 512), no imágenes promediadas
    assert vis.window_shapes and all(s == (vis.window, 512) for s in vis.window_shapes)
    assert "_meanProbs" not in j
    assert not [d for d in os.listdir(tmp_path) if d.startswith("job-")]  # sin temporales


def test_video_status_before_completion_has_null_results(make_client, mp4):
    c = make_client()
    b = c.post("/api/v1/emotions/video", files={"video": ("v.mp4", mp4, "video/mp4")}, headers=AUTH).json()
    j = c.get(b["resultUrl"], headers=AUTH).json()
    if j["status"] in ("PENDING", "PROCESSING"):
        assert j["dominantExpression"] is None and j["timeline"] is None


def test_video_failed_state(make_client, mp4):
    class Boom(FakeVisualEngine):
        def detect_faces(self, bgr):
            raise RuntimeError("fallo simulado")

    c = make_client(visual=Boom())
    b = c.post("/api/v1/emotions/video", files={"video": ("v.mp4", mp4, "video/mp4")}, headers=AUTH).json()
    j = wait_job(c, b["resultUrl"]).json()
    assert j["status"] == "FAILED" and j["error"]["code"] == "INFERENCE_ERROR"


def test_video_not_found(client):
    r = client.get("/api/v1/emotions/video/vid-nope", headers=AUTH)
    assert r.status_code == 404 and r.json()["error"]["code"] == "ANALYSIS_NOT_FOUND"


def test_video_validations(make_client, mp4, tmp_path):
    c = make_client(max_video_bytes=1000)
    assert c.post("/api/v1/emotions/video", files={"video": ("v.mp4", mp4, "video/mp4")}, headers=AUTH).status_code == 413
    c = make_client()
    assert c.post("/api/v1/emotions/video", files={"video": ("v.avi", b"RIFF" + b"\0" * 64, "video/avi")}, headers=AUTH).status_code == 415
    for bad in ("0", "6", "abc"):
        r = c.post("/api/v1/emotions/video", files={"video": ("v.mp4", mp4, "video/mp4")}, data={"sampleFps": bad}, headers=AUTH)
        assert r.status_code == 400, bad
    c2 = make_client(max_video_seconds=1)
    r = c2.post("/api/v1/emotions/video", files={"video": ("v.mp4", mp4, "video/mp4")}, headers=AUTH)
    assert r.status_code == 400
    assert not [d for d in os.listdir(tmp_path) if d.startswith("job-")]  # limpieza ante rechazo


# ---------------------------------------------------------------- multimodal
def analyze(c, img=True, **data):
    d = {"text": "Estoy muy triste", **data}
    f = {"image": ("a.jpg", jpeg_bytes(), "image/jpeg")} if img else {}
    return c.post("/api/v1/emotions/analyze", data=d, files=f, headers=AUTH)


def test_analyze_image_incongruence(make_client):
    c = make_client(text=FakeTextEngine(onehot("SADNESS", 0.8)), visual=FakeVisualEngine(static=onehot("HAPPINESS", 0.85)))
    r = analyze(c, visualWeight="0.5")
    b = r.json()
    assert r.status_code == 200 and b["modality"] == "MULTIMODAL"
    assert b["text"]["valence"] == "NEGATIVE" and b["visual"]["valence"] == "POSITIVE" and b["visual"]["faceId"] == "face-1"
    f = b["fusion"]
    assert f["method"] == "WEIGHTED_LATE_FUSION" and f["incongruence"] is True
    assert f["visualWeight"] == 0.5 and f["textWeight"] == 0.5
    assert f["probabilities"]["HAPPINESS"] == pytest.approx(0.5 * 0.85 + 0.5 * 0.2 / 6, abs=1e-3)
    assert any("engaño" in w for w in b["warnings"]) and set(b["modelVersions"]) == {"text", "visual"}
    # modalidades conservadas por separado para auditoría
    assert b["text"]["probabilities"] and b["visual"]["probabilities"]


def test_analyze_uses_default_and_custom_weight(make_client):
    c = make_client(text=FakeTextEngine(onehot("SADNESS", 0.8)), visual=FakeVisualEngine(static=onehot("HAPPINESS", 0.8)))
    assert analyze(c, visualWeight="0.9").json()["fusion"]["emotion"] == "HAPPINESS"
    assert analyze(c, visualWeight="0.1").json()["fusion"]["emotion"] == "SADNESS"
    assert analyze(c, visualWeight="1.5").status_code == 400


def test_analyze_neutral_never_incongruent(make_client):
    c = make_client(text=FakeTextEngine(onehot("NEUTRAL", 0.95)), visual=FakeVisualEngine(static=onehot("HAPPINESS", 0.95)))
    assert analyze(c).json()["fusion"]["incongruence"] is False


@pytest.mark.parametrize("faces", [0, 2])
def test_analyze_face_count_422(make_client, faces):
    c = make_client(visual=FakeVisualEngine(faces=faces))
    r = analyze(c)
    assert r.status_code == 422 and r.json()["error"]["code"] == "FACE_COUNT_NOT_SUPPORTED_FOR_FUSION"


def test_analyze_requires_exactly_one_visual_source(client, mp4):
    assert analyze(client, img=False).status_code == 400
    r = client.post("/api/v1/emotions/analyze", data={"text": "hola"},
                    files={"image": ("a.jpg", jpeg_bytes(), "image/jpeg"), "video": ("v.mp4", mp4, "video/mp4")}, headers=AUTH)
    assert r.status_code == 400 and r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_analyze_text_validation(client):
    assert analyze(client, text="").status_code == 400
    assert analyze(client, language="en").status_code == 422


def test_analyze_video_async(make_client, mp4):
    c = make_client(text=FakeTextEngine(onehot("SADNESS", 0.8)), visual=FakeVisualEngine(temporal=onehot("HAPPINESS", 0.85)))
    r = c.post("/api/v1/emotions/analyze", data={"text": "triste", "visualWeight": "0.5"},
               files={"video": ("v.mp4", mp4, "video/mp4")}, headers=AUTH)
    assert r.status_code == 202
    b = r.json()
    assert b["resultUrl"] == f"/api/v1/emotions/analyze/{b['analysisId']}"
    j = wait_job(c, b["resultUrl"]).json()
    assert j["status"] == "COMPLETED" and j["modality"] == "MULTIMODAL" and j["fusion"]["incongruence"] is True
    assert j["visual"]["temporalModel"]["type"] == "LSTM"


def test_analyze_video_multiple_faces_422(make_client, mp4, tmp_path):
    c = make_client(visual=FakeVisualEngine(faces=2))
    r = c.post("/api/v1/emotions/analyze", data={"text": "hola"}, files={"video": ("v.mp4", mp4, "video/mp4")}, headers=AUTH)
    assert r.status_code == 422 and r.json()["error"]["code"] == "FACE_COUNT_NOT_SUPPORTED_FOR_FUSION"
    assert not [d for d in os.listdir(tmp_path) if d.startswith("job-")]


def test_analyze_unknown_id_404(client):
    assert client.get("/api/v1/emotions/analyze/ana-nope", headers=AUTH).status_code == 404
    assert client.get("/api/v1/emotions/video/ana-nope", headers=AUTH).status_code == 404  # los espacios de id no se mezclan
