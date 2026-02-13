# src/sentencize.py
from __future__ import annotations

from typing import List
import re
import spacy

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm", disable=["tagger", "ner", "lemmatizer"])
        except Exception:
            # fallback if model isn't installed
            _nlp = spacy.blank("en")
        if "sentencizer" not in _nlp.pipe_names:
            _nlp.add_pipe("sentencizer")
    return _nlp


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # Normalize common TEI artifacts where structured labels are glued to content,
    # e.g. "ConclusionAcute..." -> "Conclusion Acute..."
    text = re.sub(
        r"\b(Background|Objective|Objectives|Methods|Results|Conclusion|Conclusions|Purpose|Introduction|Observations|Lessons)([A-Z])",
        r"\1 \2",
        text,
    )
    nlp = get_nlp()
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text and s.text.strip()]
