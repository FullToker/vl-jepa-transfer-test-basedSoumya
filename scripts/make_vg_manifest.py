#!/usr/bin/env python3
"""Convert Visual Genome to VL-JEPA pretrain JSONL manifest.

Usage:
    python scripts/make_vg_manifest.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from urllib.parse import urlparse

VG_DIR = Path("./data/visual_genome")
OUT_DIR = Path("./data/visual_genome")
SEED = 42

SPATIAL_PREDICATES = {
    "left of", "to the left of", "right of", "to the right of",
    "above", "below", "under", "underneath", "on top of", "on top",
    "in front of", "behind", "next to", "beside", "near", "between",
    "inside", "inside of", "outside", "over", "across from",
}


def build_image_map() -> dict[int, str]:
    """image_id -> absolute local path, only for images on disk."""
    data = json.load(open(VG_DIR / "image_data.json"))
    mapping = {}
    for item in data:
        parsed = urlparse(item["url"])
        parts = parsed.path.split("/")
        local = VG_DIR / parts[-2] / parts[-1]
        if local.exists():
            mapping[item["image_id"]] = str(local)
    print(f"images on disk: {len(mapping):,}")
    return mapping


def convert_relationships(image_map: dict) -> list[dict]:
    data = json.load(open(VG_DIR / "relationships.json"))
    samples = []
    for item in data:
        img = image_map.get(item["image_id"])
        if img is None:
            continue
        for rel in item["relationships"]:
            pred = rel.get("predicate", "").lower().strip()
            if not any(s in pred for s in SPATIAL_PREDICATES):
                continue
            subj = (rel["subject"].get("names") or [rel["subject"].get("name", "")])[0]
            obj = (rel["object"].get("names") or [rel["object"].get("name", "")])[0]
            if not subj or not obj:
                continue
            samples.append({
                "image": img,
                "query": f"What is {pred} the {obj.lower()}?",
                "target": subj.lower(),
            })
    print(f"relationships samples: {len(samples):,}")
    return samples


def convert_qa(image_map: dict) -> list[dict]:
    data = json.load(open(VG_DIR / "question_answers.json"))
    samples = []
    for item in data:
        img = image_map.get(item["id"])
        if img is None:
            continue
        for qa in item["qas"]:
            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip()
            if q and a:
                samples.append({"image": img, "query": q, "target": a})
    print(f"qa samples: {len(samples):,}")
    return samples


def write_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"wrote {len(samples):,} lines -> {path}")


if __name__ == "__main__":
    random.seed(SEED)
    image_map = build_image_map()

    rel = convert_relationships(image_map)
    write_jsonl(rel, OUT_DIR / "vg_relationships.jsonl")

    qa = convert_qa(image_map)
    write_jsonl(qa, OUT_DIR / "vg_qa.jsonl")

    combined = rel + qa
    random.shuffle(combined)
    write_jsonl(combined, OUT_DIR / "vg_pretrain.jsonl")
