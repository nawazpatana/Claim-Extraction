from __future__ import annotations

import re
from typing import Optional

# Label scheme: 1 = CLAIM, 0 = NOT_CLAIM, -1 = ABSTAIN
ABSTAIN = -1
CLAIM = 1
NOT_CLAIM = 0


_CLAIM_PATTERNS = [
    r"\b(we|this study|our results|our findings)\b.*\b(show|demonstrate|indicate|suggest|reveal)\b",
    r"\b(results|findings)\b.*\b(show|demonstrate|indicate|suggest)\b",
    r"\b(improve|outperform|achieve|increase|reduce)\b.*\b(accuracy|auc|f1|precision|recall|performance)\b",
    r"\b(significant|significantly)\b",
    r"\b(we propose|we present|we introduce)\b",
    r"\b(can|could|may)\b.*\b(help|enable|support|facilitate)\b",
]

_NONCLAIM_PATTERNS = [
    r"\b(in this paper|in this work)\b.*\b(we describe|we review|we survey)\b",
    r"\b(background|motivation|objective)\b",
    r"\b(dataset|data set|cohort)\b.*\b(consists of|comprises|contains)\b",
]

def lf_claim_pattern(sentence: str) -> int:
    s = sentence.lower()
    for p in _CLAIM_PATTERNS:
        if re.search(p, s):
            return CLAIM
    return ABSTAIN

def lf_nonclaim_pattern(sentence: str) -> int:
    s = sentence.lower()
    for p in _NONCLAIM_PATTERNS:
        if re.search(p, s):
            return NOT_CLAIM
    return ABSTAIN

def weak_claim_score(sentence: str) -> float:
    """
    Simple score in [0, 1] based on heuristics.
    Used as a fallback if SciBERT model isn't available.
    """
    votes = []
    v1 = lf_claim_pattern(sentence)
    v2 = lf_nonclaim_pattern(sentence)
    if v1 != ABSTAIN:
        votes.append(v1)
    if v2 != ABSTAIN:
        votes.append(v2)

    if not votes:
        return 0.35  # neutral prior

    # if any NOT_CLAIM vote, reduce score
    if NOT_CLAIM in votes and CLAIM not in votes:
        return 0.10
    if CLAIM in votes and NOT_CLAIM not in votes:
        return 0.80
    # conflict
    return 0.45
