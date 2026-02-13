from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_PRETRAINED = "allenai/scibert_scivocab_uncased"


def load_tokenizer(model_name: str = DEFAULT_PRETRAINED):
    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name: str = DEFAULT_PRETRAINED, num_labels: int = 2):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


@dataclass
class ClaimModel:
    model_name_or_path: str
    device: str = "cpu"

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        self.model.eval()
        self.model.to(self.device)

        # assume label space: [0=not_claim, 1=claim]
        self.claim_index = 1 if self.model.config.num_labels > 1 else 0

    @torch.inference_mode()
    def predict_proba_claim(self, sentences: List[str], batch_size: int = 16) -> List[float]:
        if not sentences:
            return []
        probs_all: List[float] = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**enc).logits
            if logits.shape[-1] == 1:
                # single-logit => sigmoid probability
                p = torch.sigmoid(logits).squeeze(-1)
                probs_all.extend(p.detach().cpu().tolist())
            else:
                p = torch.softmax(logits, dim=-1)[:, self.claim_index]
                probs_all.extend(p.detach().cpu().tolist())

        return probs_all


def try_load_claim_model(model_path: Optional[str], device: Optional[str] = None) -> Optional[ClaimModel]:
    if not model_path:
        return None
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return ClaimModel(model_name_or_path=model_path, device=device)
    except Exception:
        return None
