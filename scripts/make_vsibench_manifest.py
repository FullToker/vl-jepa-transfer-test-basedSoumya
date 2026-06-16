#!/usr/bin/env python3
"""Convert VSI-Bench test.jsonl to VL-JEPA inference manifest."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vsi-dir", default="./data/VSI-Bench")
    parser.add_argument("--out", default="data/vsibench_manifest.jsonl")
    parser.add_argument("--mode", choices=["mc", "open", "all"], default="mc",
                        help="mc=multiple-choice only, open=open-ended only, all=both")
    args = parser.parse_args()

    vsi_dir = Path(args.vsi_dir)
    src = vsi_dir / "test.jsonl"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written, skipped = 0, 0
    with open(src, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            has_options = bool(row.get("options"))

            if args.mode == "mc" and not has_options:
                continue
            if args.mode == "open" and has_options:
                continue

            video_path = vsi_dir / row["dataset"] / f"{row['scene_name']}.mp4"
            if not video_path.exists():
                skipped += 1
                continue

            if has_options:
                # Map ground truth letter (A/B/C/D) to full option string
                letter = row["ground_truth"].strip().upper()
                target = next(
                    (opt for opt in row["options"] if opt.startswith(letter + ".")),
                    row["ground_truth"],
                )
                candidates = row["options"]
            else:
                target = row["ground_truth"]
                candidates = None

            record = {
                "id": row["id"],
                "video": str(video_path),
                "query": row["question"],
                "target": target,
                "question_type": row["question_type"],
            }
            if candidates:
                record["candidates"] = candidates

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written: {written}, Skipped (missing video): {skipped}")
    print(f"Manifest saved to: {out_path}")


if __name__ == "__main__":
    main()
