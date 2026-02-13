# src/model.py
from __future__ import annotations
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SCIBERT = "allenai/scibert_scivocab_uncased"

def load_tokenizer(model_name: str = SCIBERT):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def load_model(model_name: str = SCIBERT, num_labels: int = 2):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
