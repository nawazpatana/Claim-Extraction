from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List

from labeling_functions import (
    is_incomplete_sentence,
    is_background_sentence,
    is_methods_sentence,
    is_case_observation_sentence,
)


def _norm(s: str | None) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_gold_map(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = r.get("source_file")
        if key:
            out[key] = r
    return out


def evaluate(pred_path: Path, gold_path: Path | None = None) -> Dict[str, Any]:
    preds = _load_jsonl(pred_path)
    gold_map = _build_gold_map(_load_jsonl(gold_path)) if gold_path else {}

    n = len(preds)
    with_claim = 0
    truncated_main = 0
    bg_or_methods_main = 0
    case_obs_main = 0
    strong_main = 0
    scored_gold = 0

    for r in preds:
        mc = r.get("main_claim") or ""
        if r.get("claims"):
            with_claim += 1
        if is_incomplete_sentence(mc):
            truncated_main += 1
        if is_background_sentence(mc) or is_methods_sentence(mc):
            bg_or_methods_main += 1
        if is_case_observation_sentence(mc):
            case_obs_main += 1

        key = r.get("source_file")
        g = gold_map.get(key)
        if not g:
            continue
        scored_gold += 1

        # Option 1: explicit boolean label in gold file.
        if "is_strong_main_claim" in g:
            if bool(g.get("is_strong_main_claim")):
                strong_main += 1
            continue

        # Option 2: compare predicted sentence to labeled gold main claim.
        gold_main = _norm(g.get("gold_main_claim") or g.get("main_claim") or "")
        pred_main = _norm(mc)
        if gold_main and (pred_main == gold_main or pred_main in gold_main or gold_main in pred_main):
            strong_main += 1

    out = {
        "num_rows": n,
        "claim_coverage_rate": (with_claim / n) if n else 0.0,
        "truncated_main_rate": (truncated_main / n) if n else 0.0,
        "background_or_methods_main_rate": (bg_or_methods_main / n) if n else 0.0,
        "case_observation_main_rate": (case_obs_main / n) if n else 0.0,
    }
    if scored_gold:
        out["strong_main_claim_at_1"] = strong_main / scored_gold
        out["gold_scored_rows"] = scored_gold
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate claim extraction quality and failure modes.")
    p.add_argument("--pred", required=True, help="Path to predictions.jsonl")
    p.add_argument(
        "--gold",
        default=None,
        help="Optional labeled jsonl with source_file + is_strong_main_claim or gold_main_claim",
    )
    args = p.parse_args()

    report = evaluate(Path(args.pred), Path(args.gold) if args.gold else None)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
