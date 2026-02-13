from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from tqdm import tqdm

from predict import extract_claims_from_tei
from evaluate_predictions import evaluate



def run(
    tei_dir: str,
    out_dir: str,
    model_path: str | None,
    threshold: float,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
):
    tei_dir_p = Path(tei_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    tei_files = sorted(list(tei_dir_p.glob("*.xml")))
    out_path = out_dir_p / "predictions.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for tei_path in tqdm(tei_files, desc="Processing TEI"):
            try:
                row = extract_claims_from_tei(
                    str(tei_path),
                    model_path=model_path,
                    claim_threshold=threshold,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    delta=delta,
                )
            except Exception as e:
                row = {
                    "doi": None,
                    "year": None,
                    "title": None,
                    "abstract": None,
                    "claims": [],
                    "main_claim": None,
                    "claim_method": "error",
                    "source_file": tei_path.name,
                    "error": f"{type(e).__name__}: {e}",
                }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_path}")


def build_parser():
    p = argparse.ArgumentParser(prog="claimtrace")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Extract metadata + claims from TEI XML directory")
    r.add_argument("--tei_dir", required=True, help="Directory containing TEI .xml files")
    r.add_argument("--out_dir", required=True, help="Output directory")
    r.add_argument("--model_path", default=None, help="Path to fine-tuned model (optional)")
    r.add_argument("--threshold", type=float, default=0.55, help="Claim probability threshold")
    r.add_argument("--alpha", type=float, default=0.50, help="Weight for p_claim")
    r.add_argument("--beta", type=float, default=0.20, help="Weight for result-verb indicator")
    r.add_argument("--gamma", type=float, default=0.20, help="Weight for metric indicator")
    r.add_argument("--delta", type=float, default=0.10, help="Weight for comparison indicator")

    e = sub.add_parser("eval", help="Evaluate prediction quality and failure-mode rates")
    e.add_argument("--pred", required=True, help="Path to predictions.jsonl")
    e.add_argument(
        "--gold",
        default=None,
        help="Optional labeled jsonl with source_file + is_strong_main_claim or gold_main_claim",
    )

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "run":
        run(
            args.tei_dir,
            args.out_dir,
            args.model_path,
            args.threshold,
            args.alpha,
            args.beta,
            args.gamma,
            args.delta,
        )
    elif args.cmd == "eval":
        report = evaluate(Path(args.pred), Path(args.gold) if args.gold else None)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
