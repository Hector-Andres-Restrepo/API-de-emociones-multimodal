"""Verifica que los pesos publicados de Emo-AffectNet coinciden con la arquitectura de app/engines.

Uso (dentro del contenedor o con las dependencias instaladas):
    python scripts/inspect_weights.py [ruta_backbone.pt] [ruta_lstm.pt]
Sin argumentos, descarga los archivos de Hugging Face a MODELS_DIR.
Imprime claves y formas; si hay diferencias, ajuste KEY_REMAP en app/engines/visual_engine.py.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import Settings  # noqa: E402
from app.engines.emo_affectnet_arch import EmoBackbone, EmoLSTM  # noqa: E402


def check(name, model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"\n== {name}: {path}\n   parámetros en .pt: {len(state)}  |  en modelo: {len(model.state_dict())}")
    print("   primeras claves .pt:", list(state)[:6])
    print("   primeras claves modelo:", list(model.state_dict())[:6])
    print("   FALTAN en .pt:", list(missing)[:10], "...", len(missing))
    print("   INESPERADAS en .pt:", list(unexpected)[:10], "...", len(unexpected))
    shape_bad = [(k, tuple(state[k].shape), tuple(v.shape)) for k, v in model.state_dict().items()
                 if k in state and state[k].shape != v.shape]
    print("   formas distintas:", shape_bad[:10])
    print("   RESULTADO:", "COMPATIBLE" if not (missing or unexpected or shape_bad) else "INCOMPATIBLE")


if __name__ == "__main__":
    s = Settings.from_env()
    if len(sys.argv) == 3:
        bb, lstm = sys.argv[1:3]
    else:
        from huggingface_hub import hf_hub_download
        cache = os.path.join(s.models_dir, "hf")
        bb = hf_hub_download(s.visual_repo_id, s.visual_backbone_file, cache_dir=cache)
        lstm = hf_hub_download(s.visual_repo_id, s.visual_lstm_file, cache_dir=cache)
    check("backbone", EmoBackbone(), bb)
    check("LSTM", EmoLSTM(), lstm)
