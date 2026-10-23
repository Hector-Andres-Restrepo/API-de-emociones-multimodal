"""Pruebas del núcleo: mapeo de etiquetas, normalización, fusión e incongruencia (vectores sintéticos)."""
import pytest

from app import fusion
from app.labels import (CANONICAL, TEXT_LABEL_MAP, VALENCE, VISUAL_LABEL_MAP, VISUAL_OUTPUT_ORDER, LabelMappingError,
                        align, check_mapping, normalize, top)
from tests.fakes import onehot

TEXT_REAL_LABELS = ["others", "joy", "sadness", "anger", "surprise", "disgust", "fear"]  # config.json real


def test_canonical_classes_and_valences():
    assert CANONICAL == ["ANGER", "DISGUST", "FEAR", "HAPPINESS", "NEUTRAL", "SADNESS", "SURPRISE"]
    assert VALENCE["HAPPINESS"] == "POSITIVE"
    assert all(VALENCE[c] == "NEGATIVE" for c in ["ANGER", "DISGUST", "FEAR", "SADNESS"])
    assert VALENCE["NEUTRAL"] == VALENCE["SURPRISE"] == "NEUTRAL"


def test_text_label_mapping_uses_real_names_not_indices():
    # 'others' está en el índice 0 y 'joy' en el 1: el mapeo es por nombre.
    probs = [0.05, 0.60, 0.10, 0.05, 0.05, 0.05, 0.10]
    out = align(probs, TEXT_REAL_LABELS, TEXT_LABEL_MAP)
    assert out["HAPPINESS"] == pytest.approx(0.60)
    assert out["NEUTRAL"] == pytest.approx(0.05)
    assert out["FEAR"] == pytest.approx(0.10)
    assert top(out)[0] == "HAPPINESS"


def test_visual_order_adapter_differs_from_canonical_order():
    # Orden Emo-AffectNet: Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger
    probs = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]  # índice 6 = Anger
    out = align(probs, VISUAL_OUTPUT_ORDER, VISUAL_LABEL_MAP)
    assert top(out) == ("ANGER", pytest.approx(0.9))
    assert out["NEUTRAL"] == pytest.approx(0.1)
    # un índice-a-índice ingenuo habría dado SURPRISE (posición 6 de CANONICAL)
    assert CANONICAL[6] == "SURPRISE"


def test_unknown_or_incomplete_labels_are_rejected():
    with pytest.raises(LabelMappingError):
        check_mapping(["joy", "otra"], TEXT_LABEL_MAP)
    with pytest.raises(LabelMappingError):
        check_mapping(["joy", "anger"], TEXT_LABEL_MAP)  # faltan clases
    with pytest.raises(LabelMappingError):
        align([0.5, 0.5], TEXT_REAL_LABELS, TEXT_LABEL_MAP)  # longitud distinta


def test_normalization_sums_to_one():
    p = normalize({c: i + 1.0 for i, c in enumerate(CANONICAL)})
    assert sum(p.values()) == pytest.approx(1.0)


def test_fusion_formula_matches_spec():
    pt = onehot("SADNESS", 0.8)
    pv = onehot("HAPPINESS", 0.6)
    w = 0.3
    f = fusion.fuse(pt, pv, w)
    for c in CANONICAL:
        assert f[c] == pytest.approx(w * pv[c] + (1 - w) * pt[c])
    assert sum(f.values()) == pytest.approx(1.0)
    full = fusion.fuse_full(pt, pv, w, 0.7)
    assert full["emotion"] == top(f)[0] and full["confidence"] == pytest.approx(top(f)[1])


def test_fusion_weight_extremes():
    pt, pv = onehot("SADNESS"), onehot("HAPPINESS")
    assert fusion.fuse_full(pt, pv, 0.0, 0.7)["emotion"] == "SADNESS"
    assert fusion.fuse_full(pt, pv, 1.0, 0.7)["emotion"] == "HAPPINESS"
    with pytest.raises(ValueError):
        fusion.fuse(pt, pv, 1.2)


@pytest.mark.parametrize("t,v,expected", [
    ("SADNESS", "HAPPINESS", True),
    ("HAPPINESS", "ANGER", True),
    ("FEAR", "HAPPINESS", True),
    ("HAPPINESS", "HAPPINESS", False),
    ("SADNESS", "ANGER", False),            # ambas negativas
    ("NEUTRAL", "HAPPINESS", False),         # neutral nunca activa
    ("SURPRISE", "SADNESS", False),          # sorpresa nunca activa
    ("HAPPINESS", "NEUTRAL", False),
])
def test_incongruence_valence_rules(t, v, expected):
    assert fusion.is_incongruent(onehot(t, 0.9), onehot(v, 0.9), 0.70) is expected


def test_incongruence_requires_both_confidences_at_threshold():
    assert fusion.is_incongruent(onehot("SADNESS", 0.70), onehot("HAPPINESS", 0.70), 0.70) is True
    assert fusion.is_incongruent(onehot("SADNESS", 0.69), onehot("HAPPINESS", 0.95), 0.70) is False
    assert fusion.is_incongruent(onehot("SADNESS", 0.95), onehot("HAPPINESS", 0.69), 0.70) is False
    assert fusion.is_incongruent(onehot("SADNESS", 0.69), onehot("HAPPINESS", 0.69), 0.60) is True  # umbral configurable
