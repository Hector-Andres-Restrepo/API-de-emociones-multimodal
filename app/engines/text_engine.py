"""Clasificación de emociones en texto: pysentimiento/robertuito-emotion-analysis."""
import logging
import os
from typing import Dict, List, Tuple

from ..config import Settings
from ..labels import TEXT_LABEL_MAP, align, check_mapping

log = logging.getLogger("emotions.text")


class RobertuitoEngine:
    MAX_TOKENS = 128  # max_position_embeddings=130 en config.json

    def __init__(self, settings: Settings):
        self.s = settings
        self.ready = False
        self.version = f"{settings.text_model_id}@unloaded"
        self.labels: List[str] = []

    def load(self) -> None:
        import torch  # import perezoso: la API arranca aunque falte torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        cache = os.path.join(self.s.models_dir, "hf")
        self.tokenizer = AutoTokenizer.from_pretrained(self.s.text_model_id, cache_dir=cache)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.s.text_model_id, cache_dir=cache).eval()
        id2label = self.model.config.id2label
        self.labels = [str(id2label[i]) for i in range(len(id2label))]
        check_mapping(self.labels, TEXT_LABEL_MAP)  # verificación obligatoria de etiquetas reales
        log.info("Etiquetas reales del modelo de texto: %s", self.labels)
        rev = getattr(self.model.config, "_commit_hash", None)
        self.version = f"robertuito-emotion-analysis@{rev[:8] if rev else 'configured'}"
        self._torch = torch
        try:
            from pysentimiento.preprocessing import preprocess_tweet
            self._pre = lambda t: preprocess_tweet(t, lang="es")
        except Exception:  # pragma: no cover
            log.warning("pysentimiento.preprocessing no disponible; se usa el texto sin preprocesar")
            self._pre = lambda t: t
        self.ready = True

    def predict(self, text: str) -> Tuple[Dict[str, float], List[str]]:
        t = self._pre(text)
        warnings: List[str] = []
        if len(self.tokenizer(t, truncation=False)["input_ids"]) > self.MAX_TOKENS:
            warnings.append(f"TEXT_TRUNCATED: el modelo analiza solo los primeros {self.MAX_TOKENS} tokens.")
        enc = self.tokenizer(t, return_tensors="pt", truncation=True, max_length=self.MAX_TOKENS)
        with self._torch.inference_mode():
            probs = self._torch.softmax(self.model(**enc).logits[0], dim=-1).tolist()
        return align(probs, self.labels, TEXT_LABEL_MAP), warnings
