from __future__ import annotations

from typing import List

import numpy as np

from .labeling_functions import lf_claim_pattern, lf_nonclaim_pattern

def snorkel_label_probs(sentences: List[str]) -> List[float]:
    """
    Optional: Returns P(CLAIM) per sentence using a Snorkel LabelModel.
    If snorkel isn't installed, raise a clear error.
    """
    try:
        from snorkel.labeling import LabelModel
    except Exception as e:
        raise RuntimeError(
            "Snorkel is not installed. Install with: pip install snorkel==0.9.9"
        ) from e

    if not sentences:
        return []

    L = np.zeros((len(sentences), 2), dtype=int)
    for i, s in enumerate(sentences):
        L[i, 0] = lf_claim_pattern(s)
        L[i, 1] = lf_nonclaim_pattern(s)

    label_model = LabelModel(cardinality=2, verbose=False)
    label_model.fit(L_train=L, n_epochs=200, log_freq=0, seed=13)

    probs = label_model.predict_proba(L)
    return probs[:, 1].tolist()
