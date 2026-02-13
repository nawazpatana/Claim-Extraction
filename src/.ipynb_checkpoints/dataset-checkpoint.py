# src/dataset.py
from __future__ import annotations
import pandas as pd
from datasets import Dataset

def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    # Expect columns: text, weak_y, weak_p_claim, paper_id, sent_id
    keep = ["paper_id", "sent_id", "text", "weak_y", "weak_p_claim"]
    return Dataset.from_pandas(df[keep], preserve_index=False)
