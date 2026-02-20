#!/usr/bin/env python3
from __future__ import annotations

"""Create synthetic local data for offline smoke tests."""

import json
from pathlib import Path

from PIL import Image


def main() -> None:
    root = Path("data/tiny")
    root.mkdir(parents=True, exist_ok=True)

    samples = [
        ("red.png", (220, 20, 20), "What color is this?", "red square"),
        ("green.png", (20, 180, 20), "What color is this?", "green square"),
    ]
    rows = []
    for name, rgb, query, target in samples:
        path = root / name
        Image.new("RGB", (64, 64), color=rgb).save(path)
        rows.append({"image": str(path), "query": query, "target": target})

    manifest = Path("data/tiny_pretrain_manifest.jsonl")
    with open(manifest, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    text_bank = Path("data/tiny_text_bank.txt")
    text_bank.write_text("red square\ngreen square\nblue square\n", encoding="utf-8")

    infer_manifest = Path("data/tiny_infer_manifest.jsonl")
    infer_manifest.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    print("Created tiny data and manifests.")


if __name__ == "__main__":
    main()

