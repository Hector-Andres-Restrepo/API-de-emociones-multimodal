"""Vuelca claves y formas del backbone Emo-AffectNet (solo necesita torch y huggingface_hub).

    docker compose run --rm -v "${PWD}/scripts:/app/scripts" api python scripts/dump_keys.py

Imprime un resumen compacto y guarda la lista completa en scripts/backbone_keys.txt.
"""
import os
import re

import torch
from huggingface_hub import hf_hub_download

repo = os.getenv("VISUAL_REPO_ID", "ElenaRyumina/face_emotion_recognition")
fname = os.getenv("VISUAL_BACKBONE_FILE", "FER_static_ResNet50_AffectNet.pt")
path = hf_hub_download(repo, fname, cache_dir=os.path.join(os.getenv("MODELS_DIR", "/models"), "hf"))
state = torch.load(path, map_location="cpu", weights_only=True)

lines = [f"{k}\t{tuple(v.shape)}" for k, v in state.items()]
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backbone_keys.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"TOTAL {len(state)} claves")
for k, v in state.items():
    m = re.match(r"layer(\d+)\.(\d+)\.", k)
    # resumen: todo lo que no es bloque, más los bloques 0 de cada capa y el 1 de layer1
    if m is None or m.group(2) == "0" or (m.group(1) == "1" and m.group(2) == "1"):
        if "num_batches_tracked" not in k:
            print(f"{k}\t{tuple(v.shape)}")
print("capas y bloques:", sorted({re.match(r'(layer\d+\.\d+)\.', k).group(1) for k in state if re.match(r'layer\d+\.\d+\.', k)}))
