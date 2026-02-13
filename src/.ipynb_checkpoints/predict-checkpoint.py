from __future__ import annotations

from typing import Dict, Any, List, Optional

from tei_parser import parse_tei_metadata
from sentencize import split_sentences
from model import try_load_claim_model
from labeling_functions import weak_claim_score


# optional snorkel:
# from .weak_label import snorkel_label_probs


def extract_claims_from_tei(
    tei_path: str,
    model_path: Optional[str] = None,
    claim_threshold: float = 0.55,
    top_k_claims: int = 6,
) -> Dict[str, Any]:
    meta = parse_tei_metadata(tei_path)

    abstract = meta.get("abstract") or ""
    sentences = split_sentences(abstract)

    # Choose scoring method:
    model = try_load_claim_model(model_path)
    if model:
        scores = model.predict_proba_claim(sentences)
        method = "scibert"
    else:
        # fallback: heuristics
        scores = [weak_claim_score(s) for s in sentences]
        method = "weak_rules"

        # If you want Snorkel later, replace with:
        # scores = snorkel_label_probs(sentences)
        # method = "snorkel"

    # pick claim sentences
    scored = list(zip(sentences, scores))
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)

    claims = [s for s, p in scored_sorted if p >= claim_threshold][:top_k_claims]
    main_claim = scored_sorted[0][0] if scored_sorted else None

    out = {
        "doi": meta.get("doi"),
        "year": meta.get("year"),
        "title": meta.get("title"),
        "abstract": meta.get("abstract"),
        "claims": claims,
        "main_claim": main_claim,
        "claim_method": method,
        "source_file": meta.get("source_file"),
    }
    return out
