# src/train.py
from __future__ import annotations
import numpy as np
from transformers import Trainer, TrainingArguments
from .model import load_model, load_tokenizer

def tokenize_batch(examples, tokenizer, max_len=256):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_len)

def train_scibert(hf_ds, out_dir: str, epochs: int = 3, lr: float = 2e-5, bs: int = 16):
    tokenizer = load_tokenizer()
    model = load_model()

    ds = hf_ds.map(lambda x: tokenize_batch(x, tokenizer), batched=True)
    ds = ds.rename_column("weak_y", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=lr,
        per_device_train_batch_size=bs,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir
