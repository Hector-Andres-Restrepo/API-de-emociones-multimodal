import io
import os
import time

os.environ["EMOTIONS_API_NO_AUTOAPP"] = "1"

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from tests.fakes import FakeTextEngine, FakeVisualEngine

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def _make(text=None, visual=None, **over):
        s = Settings(api_token=TOKEN, tmp_dir=str(tmp_path), models_dir=str(tmp_path / "m"), **over)
        app = create_app(s, text or FakeTextEngine(), visual or FakeVisualEngine(), load_models=False)
        c = TestClient(app)
        c.__enter__()
        clients.append(c)
        return c

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()


def jpeg_bytes(w=64, h=64) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def make_mp4(path, seconds=4, fps=10, size=(64, 64)) -> bytes:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(seconds * fps):
        vw.write(np.full((size[1], size[0], 3), (i * 3) % 255, np.uint8))
    vw.release()
    return open(path, "rb").read()


@pytest.fixture
def mp4(tmp_path):
    return make_mp4(tmp_path / "v.mp4")


def wait_job(client, url, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(url, headers=AUTH)
        if r.json().get("status") in ("COMPLETED", "FAILED"):
            return r
        time.sleep(0.1)
    raise AssertionError("el job no terminó")
