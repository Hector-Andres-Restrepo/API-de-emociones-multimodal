"""Fusión tardía de probabilidades e indicador de incongruencia.

Pesos, umbral y valencias proceden de una propuesta experimental del informe;
no son una regla científicamente validada. Todo es parametrizable.
"""
from typing import Dict

from .labels import CANONICAL, VALENCE, normalize, top


def fuse(p_text: Dict[str, float], p_visual: Dict[str, float], visual_weight: float) -> Dict[str, float]:
    if not 0.0 <= visual_weight <= 1.0:
        raise ValueError("visual_weight debe estar en [0, 1]")
    return {c: visual_weight * p_visual[c] + (1.0 - visual_weight) * p_text[c] for c in CANONICAL}


def valence_of(emotion: str) -> str:
    return VALENCE[emotion]


def is_incongruent(p_text: Dict[str, float], p_visual: Dict[str, float], threshold: float) -> bool:
    """True solo si las clases predominantes tienen valencias POSITIVE vs NEGATIVE
    y ambas confianzas (prob. máxima de cada modalidad) alcanzan el umbral.
    NEUTRAL y SURPRISE (valencia NEUTRAL) nunca activan la oposición."""
    t_emo, t_conf = top(p_text)
    v_emo, v_conf = top(p_visual)
    opposed = {valence_of(t_emo), valence_of(v_emo)} == {"POSITIVE", "NEGATIVE"}
    return bool(opposed and t_conf >= threshold and v_conf >= threshold)


def fuse_full(p_text, p_visual, visual_weight: float, threshold: float) -> dict:
    p = normalize(fuse(p_text, p_visual, visual_weight))
    emotion, conf = top(p)
    return {
        "probabilities": p,
        "emotion": emotion,
        "confidence": conf,
        "incongruence": is_incongruent(p_text, p_visual, threshold),
    }
