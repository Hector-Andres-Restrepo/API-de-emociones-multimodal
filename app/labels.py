"""Clases canónicas, valencias y adaptadores explícitos de etiquetas.

Nunca se asume coincidencia de índices entre modelos: cada adaptador recibe
las etiquetas REALES del modelo (leídas en la carga) y las mapea por nombre.
"""
from typing import Dict, List, Sequence, Tuple

CANONICAL: List[str] = ["ANGER", "DISGUST", "FEAR", "HAPPINESS", "NEUTRAL", "SADNESS", "SURPRISE"]

VALENCE: Dict[str, str] = {
    "HAPPINESS": "POSITIVE",
    "ANGER": "NEGATIVE",
    "DISGUST": "NEGATIVE",
    "FEAR": "NEGATIVE",
    "SADNESS": "NEGATIVE",
    "NEUTRAL": "NEUTRAL",
    "SURPRISE": "NEUTRAL",
}

# RoBERTuito (config.json verificado): id2label = {0: others, 1: joy, 2: sadness,
# 3: anger, 4: surprise, 5: disgust, 6: fear}. 'joy' -> HAPPINESS; 'others' -> NEUTRAL
# (clase residual del corpus TASS 2020/EmoEvent; supuesto documentado en el informe).
TEXT_LABEL_MAP: Dict[str, str] = {
    "anger": "ANGER",
    "disgust": "DISGUST",
    "fear": "FEAR",
    "joy": "HAPPINESS",
    "sadness": "SADNESS",
    "surprise": "SURPRISE",
    "others": "NEUTRAL",
}

# Emo-AffectNet (run_webcam.ipynb oficial): DICT_EMO = {0: Neutral, 1: Happiness,
# 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger}
VISUAL_OUTPUT_ORDER: List[str] = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger"]
VISUAL_LABEL_MAP: Dict[str, str] = {n: n.upper() for n in VISUAL_OUTPUT_ORDER}


class LabelMappingError(ValueError):
    pass


def check_mapping(source_labels: Sequence[str], label_map: Dict[str, str]) -> None:
    """Falla si las etiquetas reales del modelo no cubren exactamente las 7 canónicas."""
    unknown = [l for l in source_labels if l not in label_map]
    if unknown:
        raise LabelMappingError(f"Etiquetas del modelo sin mapeo canónico: {unknown}")
    mapped = sorted(label_map[l] for l in source_labels)
    if mapped != sorted(CANONICAL):
        raise LabelMappingError(f"El mapeo no produce las 7 clases canónicas: {mapped}")


def align(probs: Sequence[float], source_labels: Sequence[str], label_map: Dict[str, str]) -> Dict[str, float]:
    """Reordena un vector de probabilidades (en el orden REAL del modelo) al diccionario canónico."""
    if len(probs) != len(source_labels):
        raise LabelMappingError(f"{len(probs)} probabilidades vs {len(source_labels)} etiquetas")
    check_mapping(source_labels, label_map)
    out = {c: 0.0 for c in CANONICAL}
    for p, lab in zip(probs, source_labels):
        out[label_map[lab]] += float(p)
    return normalize(out)


def normalize(probs: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in probs.values())
    if total <= 0:
        raise LabelMappingError("Vector de probabilidades con suma no positiva")
    return {c: max(probs.get(c, 0.0), 0.0) / total for c in CANONICAL}


def top(probs: Dict[str, float]) -> Tuple[str, float]:
    emotion = max(CANONICAL, key=lambda c: probs[c])
    return emotion, probs[emotion]


def rounded(probs: Dict[str, float], nd: int = 4) -> Dict[str, float]:
    return {c: round(probs[c], nd) for c in CANONICAL}
