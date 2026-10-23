"""Pruebas unitarias de LiveAnalyzer (lógica de la sesión multimodal en vivo) con reloj simulado.
Los motores simulados viven en tests/fakes.py y se usan SOLO aquí, nunca en la aplicación real."""
import logging
from collections import deque

import numpy as np
import pytest

from app.config import Settings
from app.deps import DISCLAIMER_FACE, DISCLAIMER_INCONGRUENCE, DISCLAIMER_TEXT
from app.errors import ApiError
from app.labels import CANONICAL
from app.services.live import (PARTIAL_TEXT_NOTE, STATUS_FACE_COUNT, STATUS_OK, STATUS_WAITING_FACE,
                               STATUS_WAITING_TEXT, LiveAnalyzer)
from tests.fakes import FakeVisualEngine, KeywordTextEngine, onehot

IMG = np.zeros((64, 64, 3), np.uint8)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def make(text=None, visual=None, weight=0.5, window=5.0, fps=2.0, **settings):
    clock = Clock()
    s = Settings(api_token="t", **settings)
    vis = visual or FakeVisualEngine()
    an = LiveAnalyzer(text or KeywordTextEngine(), vis, s, sample_fps=fps, visual_weight=weight,
                      visual_window_s=window, clock=clock)
    return an, vis, clock


def contains(obj, needle, depth=0) -> bool:
    if depth > 6:
        return False
    if isinstance(obj, (str, bytes)):
        return needle in obj if isinstance(obj, str) else needle.encode() in obj
    if isinstance(obj, dict):
        return any(contains(k, needle, depth + 1) or contains(v, needle, depth + 1) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set, deque)):
        return any(contains(v, needle, depth + 1) for v in obj)
    return False


# ------------------------------------------------------------------ estados de fusión
def test_frames_without_text_wait_for_text():
    an, vis, _ = make(visual=FakeVisualEngine(temporal=onehot("HAPPINESS", 0.85)))
    u = an.on_frame(IMG)
    assert u["type"] == "update" and u["trigger"] == "FRAME" and u["facesDetected"] == 1
    assert u["fusionStatus"] == STATUS_WAITING_TEXT and u["fusion"] is None and u["text"] is None
    assert u["visual"]["emotion"] == "HAPPINESS" and u["visual"]["mode"] == "LSTM" and u["visual"]["faceId"] == "face-1"
    assert u["visual"]["temporal"] == {"windowFrames": vis.window, "windowFilled": vis.window}


def test_text_without_frames_waits_for_face():
    an, _, _ = make()
    u = an.on_text("Estoy muy triste", 1)
    assert u["trigger"] == "TEXT" and u["fusionStatus"] == STATUS_WAITING_FACE
    assert u["fusion"] is None and u["visual"] is None and u["facesDetected"] is None
    assert u["text"]["emotion"] == "SADNESS" and u["text"]["valence"] == "NEGATIVE" and u["text"]["textVersion"] == 1
    assert set(u["text"]["probabilities"]) == set(CANONICAL) and u["text"]["chars"] == len("Estoy muy triste")


def test_fusion_matches_formula_and_flags_incongruence():
    an, _, _ = make(visual=FakeVisualEngine(temporal=onehot("HAPPINESS", 0.85)), weight=0.5)
    an.on_text("Estoy muy triste")
    u = an.on_frame(IMG)
    f = u["fusion"]
    assert u["fusionStatus"] == STATUS_OK and f["method"] == "WEIGHTED_LATE_FUSION"
    assert f["incongruence"] is True and f["incongruenceThreshold"] == 0.70
    assert f["visualWeight"] == 0.5 and f["textWeight"] == 0.5
    expected_happy = 0.5 * 0.85 + 0.5 * (0.2 / 6)
    assert f["probabilities"]["HAPPINESS"] == pytest.approx(expected_happy, abs=1e-3)
    assert sum(f["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert f["emotion"] == "HAPPINESS" and f["confidence"] == pytest.approx(expected_happy, abs=1e-3)
    # se conservan por separado para auditoría
    assert u["text"]["emotion"] == "SADNESS" and u["visual"]["emotion"] == "HAPPINESS"


def test_editing_text_recomputes_fusion_without_new_frames():
    an, vis, _ = make(visual=FakeVisualEngine(temporal=onehot("HAPPINESS", 0.85)))
    an.on_frame(IMG)
    sad = an.on_text("Estoy muy triste")
    happy = an.on_text("Estoy muy feliz")
    assert sad["fusion"]["incongruence"] is True
    assert happy["trigger"] == "TEXT" and happy["fusion"]["incongruence"] is False
    assert happy["fusion"]["emotion"] == "HAPPINESS"
    assert len(vis.window_shapes) == 1  # ningún fotograma nuevo


def test_empty_text_clears_the_text_and_fusion():
    an, _, _ = make()
    an.on_frame(IMG)
    an.on_text("Estoy muy triste")
    u = an.on_text("   ")
    assert u["text"] is None and u["fusion"] is None and u["fusionStatus"] == STATUS_WAITING_TEXT
    assert u["visual"] is not None


# ------------------------------------------------------------------ ventana visual reciente
def test_visual_component_is_window_mean_and_expires():
    vis = FakeVisualEngine(temporal=onehot("HAPPINESS", 0.9))
    an, _, clock = make(visual=vis, window=5.0)
    an.on_text("Estoy tranquilo")
    an.on_frame(IMG)                                  # t=0: alegría
    clock.advance(1.0)
    vis.temporal_probs = onehot("NEUTRAL", 0.9)
    u = an.on_frame(IMG)                              # t=1: neutral
    assert u["visual"]["framesUsed"] == 2 and u["visual"]["windowSeconds"] == 5.0
    assert u["visual"]["probabilities"]["HAPPINESS"] == pytest.approx((0.9 + 0.1 / 6) / 2, abs=1e-3)
    assert u["visual"]["probabilities"]["NEUTRAL"] == pytest.approx((0.9 + 0.1 / 6) / 2, abs=1e-3)
    clock.advance(4.5)                                # t=5.5: el de t=0 sale de la ventana
    u = an.set_visual_weight(0.5)
    assert u["visual"]["framesUsed"] == 1 and u["visual"]["emotion"] == "NEUTRAL"
    assert u["visual"]["lastFrameAgeMs"] == 4500
    clock.advance(2.0)                                # t=7.5: ya no queda nada
    u = an.set_visual_weight(0.5)
    assert u["visual"] is None and u["fusion"] is None and u["fusionStatus"] == STATUS_WAITING_FACE


def test_lstm_receives_full_window_vectors():
    an, vis, clock = make()
    for _ in range(3):
        an.on_frame(IMG)
        clock.advance(0.5)
    assert vis.window_shapes and all(sh == (vis.window, 512) for sh in vis.window_shapes)


# ------------------------------------------------------------------ rostros
def test_multiple_faces_block_fusion_and_skip_temporal_window():
    vis = FakeVisualEngine(faces=2)
    an, _, _ = make(visual=vis)
    an.on_text("Estoy muy triste")
    u = an.on_frame(IMG)
    assert u["facesDetected"] == 2 and u["fusion"] is None and u["fusionStatus"] == STATUS_FACE_COUNT
    assert any(w.startswith("MULTIPLE_FACES_DETECTED") for w in u["warnings"])
    assert vis.window_shapes == [] and u["visual"] is None   # el LSTM no se alimenta con varios rostros
    vis.faces = 1
    u = an.on_frame(IMG)
    assert u["fusionStatus"] == STATUS_OK and u["fusion"] is not None


def test_no_face_uses_recent_frames_then_waits():
    vis = FakeVisualEngine()
    an, _, clock = make(visual=vis, window=5.0)
    an.on_text("Estoy muy triste")
    an.on_frame(IMG)
    clock.advance(1.0)
    vis.faces = 0
    u = an.on_frame(IMG)
    assert u["facesDetected"] == 0 and "NO_FACE_DETECTED" in u["warnings"]
    assert u["fusionStatus"] == STATUS_OK and u["visual"]["framesUsed"] == 1  # usa lo reciente, no inventa nada nuevo
    clock.advance(6.0)
    u = an.on_frame(IMG)
    assert u["fusionStatus"] == STATUS_WAITING_FACE and u["fusion"] is None and u["visual"] is None


# ------------------------------------------------------------------ incongruencia
@pytest.mark.parametrize("text_word,visual_emo,expected", [
    ("triste", "HAPPINESS", True),
    ("feliz", "ANGER", True),
    ("feliz", "HAPPINESS", False),
    ("triste", "ANGER", False),          # ambas negativas
    ("tranquilo", "HAPPINESS", False),   # neutral no activa
    ("feliz", "NEUTRAL", False),
    ("sorpresa", "SADNESS", False),      # sorpresa no activa
])
def test_incongruence_rules_in_live(text_word, visual_emo, expected):
    an, _, _ = make(visual=FakeVisualEngine(temporal=onehot(visual_emo, 0.9)))
    an.on_frame(IMG)
    u = an.on_text(f"Me siento {text_word}")
    assert u["fusion"]["incongruence"] is expected


def test_incongruence_needs_both_confidences_and_threshold_is_configurable():
    low_text = {"triste": onehot("SADNESS", 0.65)}
    an, _, _ = make(text=KeywordTextEngine(low_text), visual=FakeVisualEngine(temporal=onehot("HAPPINESS", 0.95)))
    an.on_frame(IMG)
    assert an.on_text("triste")["fusion"]["incongruence"] is False          # 0.65 < 0.70
    an2, _, _ = make(text=KeywordTextEngine(low_text), visual=FakeVisualEngine(temporal=onehot("HAPPINESS", 0.95)),
                     incongruence_threshold=0.60)
    an2.on_frame(IMG)
    u = an2.on_text("triste")
    assert u["fusion"]["incongruence"] is True and u["fusion"]["incongruenceThreshold"] == 0.60


# ------------------------------------------------------------------ peso visual
def test_visual_weight_can_change_during_the_session():
    an, _, _ = make(visual=FakeVisualEngine(temporal=onehot("HAPPINESS", 0.85)))
    an.on_frame(IMG)
    an.on_text("Estoy muy triste")
    u = an.set_visual_weight(1)
    assert u["trigger"] == "CONFIG" and u["fusion"]["emotion"] == "HAPPINESS"
    assert u["fusion"]["visualWeight"] == 1.0 and u["fusion"]["textWeight"] == 0.0
    u = an.set_visual_weight(0)
    assert u["fusion"]["emotion"] == "SADNESS" and u["fusion"]["textWeight"] == 1.0


@pytest.mark.parametrize("bad", [1.5, -0.1, "0.5", None, True, [0.5]])
def test_invalid_visual_weight_is_rejected_and_keeps_previous(bad):
    an, _, _ = make(weight=0.3)
    with pytest.raises(ApiError) as e:
        an.set_visual_weight(bad)
    assert e.value.code == "VALIDATION_ERROR" and an.visual_weight == 0.3


# ------------------------------------------------------------------ texto
def test_same_text_is_not_reclassified():
    text = KeywordTextEngine()
    an, _, _ = make(text=text)
    an.on_text("Estoy triste")
    an.on_text("Estoy triste")
    assert text.n_calls == 1 and an.texts_analyzed == 1
    an.on_text("Estoy triste hoy")
    assert text.n_calls == 2 and an.texts_analyzed == 2


@pytest.mark.parametrize("bad", [None, 5, ["a"], {"t": 1}])
def test_text_must_be_a_string(bad):
    an, _, _ = make()
    with pytest.raises(ApiError) as e:
        an.on_text(bad)
    assert e.value.code == "VALIDATION_ERROR" and e.value.details["field"] == "text"


def test_text_length_limit():
    an, _, _ = make(max_text_chars=10)
    an.on_text("x" * 10)
    with pytest.raises(ApiError) as e:
        an.on_text("x" * 11)
    assert e.value.code == "VALIDATION_ERROR"


# ------------------------------------------------------------------ cadencia
def test_text_cadence_returns_retry_after():
    an, _, clock = make(live_text_min_interval_ms=300)
    assert an.check_text_rate() == 0
    clock.advance(0.1)
    wait = an.check_text_rate()
    assert 150 <= wait <= 200
    clock.advance(0.21)
    assert an.check_text_rate() == 0


def test_frame_cadence_follows_sample_fps():
    an, _, clock = make(fps=1.0)
    assert an.check_frame_rate() is True
    clock.advance(0.3)
    assert an.check_frame_rate() is False
    clock.advance(0.55)           # 0.85 s desde el primero: > 0.8 s de tolerancia
    assert an.check_frame_rate() is True


# ------------------------------------------------------------------ privacidad y advertencias
def test_text_content_is_not_retained_or_logged(caplog):
    secret = "mensaje-secreto-muy-personal-98765"
    an, _, _ = make()
    with caplog.at_level(logging.DEBUG):
        an.on_frame(IMG)
        u = an.on_text(secret)
    own_state = {k: v for k, v in vars(an).items() if k not in ("text_engine", "visual", "s", "clock")}
    assert not contains(own_state, secret)
    assert not contains(u, secret) and secret not in caplog.text
    assert set(an.summary()) == {"framesProcessed", "textsAnalyzed", "fusionsComputed", "incongruentFusions"}


def test_scope_warnings_are_present_with_fusion():
    an, _, _ = make()
    an.on_frame(IMG)
    u = an.on_text("Estoy muy triste")
    w = u["warnings"]
    assert DISCLAIMER_FACE in w and DISCLAIMER_TEXT in w and DISCLAIMER_INCONGRUENCE in w and PARTIAL_TEXT_NOTE in w
    assert len(w) == len(set(w))


def test_summary_counters():
    an, _, clock = make(visual=FakeVisualEngine(temporal=onehot("HAPPINESS", 0.9)))
    an.on_text("Estoy triste")           # sin rostro: sin fusión
    clock.advance(0.5)
    an.on_frame(IMG)                     # fusión 1 (incongruente)
    an.on_text("Estoy feliz")            # fusión 2 (congruente)
    assert an.summary() == {"framesProcessed": 1, "textsAnalyzed": 2, "fusionsComputed": 2, "incongruentFusions": 1}
