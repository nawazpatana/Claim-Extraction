from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from tqdm import tqdm

from predict import extract_claims_from_tei



def run(tei_dir: str, out_dir: str, model_path: str | None, threshold: float):
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

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "run":
        run(args.tei_dir, args.out_dir, args.model_path, args.threshold)


if __name__ == "__main__":
    main()
