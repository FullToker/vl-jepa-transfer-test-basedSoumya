#!/usr/bin/env python3
"""Download VSI-Bench from HuggingFace and generate inference manifest."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vljepa.utils.hf_setup import setup_hf


REPO_ID = "nyu-visionx/VSI-Bench"
VIDEO_ZIPS = ["arkitscenes.zip", "scannet.zip", "scannetpp.zip"]


def download(dest: Path) -> None:
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)

    annotation_files = [
        "test.jsonl",
        "test_debiased.parquet",
        "test_pruned.parquet",
        "pruned_ids.txt",
    ]
    print("==> Downloading annotations...")
    for fname in annotation_files:
        out = dest / fname
        if out.exists():
            print(f"  skip (exists): {fname}")
            continue
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=fname,
            local_dir=str(dest),
        )
        print(f"  downloaded: {fname}")

    print("==> Downloading videos (~large)...")
    for fname in VIDEO_ZIPS:
        zip_path = dest / fname
        extracted_dir = dest / fname.replace(".zip", "")
        if extracted_dir.exists():
            print(f"  skip (already extracted): {fname}")
            continue
        if not zip_path.exists():
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=fname,
                local_dir=str(dest),
            )
            print(f"  downloaded: {fname}")
        print(f"  extracting: {fname} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        zip_path.unlink()
        print(f"  extracted and removed zip: {fname}")


def make_manifest(dest: Path, out: Path, mode: str) -> None:
    src = dest / "test.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with open(src) as fin, open(out, "w") as fout:
        for line in fin:
            row = json.loads(line)
            has_options = bool(row.get("options"))

            if mode == "mc" and not has_options:
                continue
            if mode == "open" and has_options:
                continue

            video_path = dest / row["dataset"] / f"{row['scene_name']}.mp4"
            if not video_path.exists():
                skipped += 1
                continue

            if has_options:
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

    print(f"Manifest: {written} written, {skipped} skipped (missing video) -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VSI-Bench and build manifest")
    parser.add_argument("--dest", default="./data/VSI-Bench")
    parser.add_argument("--manifest-out", default="data/vsibench_manifest.jsonl")
    parser.add_argument("--mode", choices=["mc", "open", "all"], default="mc")
    args = parser.parse_args()

    setup_hf()

    dest = Path(args.dest)
    download(dest)
    make_manifest(dest, Path(args.manifest_out), args.mode)
    print("==> Done.")


if __name__ == "__main__":
    main()
