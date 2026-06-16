#!/usr/bin/env python3
from __future__ import annotations

"""VL-JEPA inference entrypoint for captioning, discriminative VQA, selective decoding."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vljepa.data.dataset import VisionLanguageJsonlDataset, vl_collate
from vljepa.eval.tasks import discriminative_match
from vljepa.inference.decoder import NearestNeighborDecoder
from vljepa.train.build import build_model
from vljepa.utils.config import load_yaml, validate_train_config


def main() -> None:
    parser = argparse.ArgumentParser(description="VL-JEPA inference")
    parser.add_argument("--config", type=str, default="vljepa/configs/inference.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True, help="JSONL samples")
    parser.add_argument("--text-bank", type=str, default=None, help="Optional text bank txt/jsonl for NN decoder")
    parser.add_argument("--mode", type=str, choices=["caption", "discriminative_vqa", "selective"], default="caption")
    parser.add_argument("--num-segments", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    validate_train_config(cfg, require_runtime=True, require_train=False)
    device = torch.device(cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu")

    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).float().eval()

    ds = VisionLanguageJsonlDataset(
        manifest_path=args.manifest,
        num_frames=cfg["data"]["num_frames"],
        image_size=cfg["data"]["image_size"],
    )

    if args.mode == "caption":
        # Caption mode uses lightweight nearest-neighbor readout in text-embedding bank.
        if args.text_bank is None:
            raise ValueError("--text-bank is required for caption mode.")
        text_bank = []
        with open(args.text_bank, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    text_bank.append(json.loads(line)["text"] if line.startswith("{") else line)
        decoder = NearestNeighborDecoder(model, text_bank)
        decoder.build(device)
        loader = DataLoader(
            ds, batch_size=args.batch_size, num_workers=args.num_workers,
            collate_fn=vl_collate, pin_memory=True,
        )
        idx = 0
        for batch in tqdm(loader, desc="caption"):
            with torch.no_grad():
                preds = model.predict_embedding(batch["frames"].to(device), batch["query"])
            texts = decoder.decode(preds, topk=1)
            for text in texts:
                print(json.dumps({"idx": idx, "prediction": text}, ensure_ascii=False), flush=True)
                idx += 1

    elif args.mode == "discriminative_vqa":
        loader = DataLoader(
            ds, batch_size=args.batch_size, num_workers=args.num_workers,
            collate_fn=vl_collate, pin_memory=True,
        )
        idx = 0
        for batch in tqdm(loader, desc="discriminative_vqa"):
            if not batch["candidates"][0]:
                raise ValueError("Sample missing `candidates` for discriminative mode.")
            with torch.no_grad():
                preds = model.predict_embedding(batch["frames"].to(device), batch["query"])
            answers = discriminative_match(model, preds, batch["candidates"])
            for ans in answers:
                print(json.dumps({"idx": idx, "prediction": ans}, ensure_ascii=False), flush=True)
                idx += 1

    else:
        from vljepa.inference.selective import selective_decode_points

        # Selective decoding mode follows paper Sec. 4.6:
        # segment the embedding stream, decode only representative points.
        if len(ds) != 1:
            raise ValueError("Selective mode expects a single sample video in manifest.")
        batch = vl_collate([ds[0]])
        frames = batch["frames"].to(device)[0]  # [T, C, H, W]
        query = batch["query"][0]
        per_t = []
        for t in range(frames.shape[0]):
            emb = model.predict_embedding(frames[t : t + 1].unsqueeze(0), [query])[0]
            per_t.append(emb)
        stream = torch.stack(per_t, dim=0)
        idx, emb = selective_decode_points(stream, num_segments=args.num_segments, use_avg_pool=True)
        print(json.dumps({"decode_indices": idx, "num_points": len(idx)}))
        if args.text_bank is not None:
            text_bank = [x.strip() for x in Path(args.text_bank).read_text(encoding="utf-8").splitlines() if x.strip()]
            decoder = NearestNeighborDecoder(model, text_bank)
            decoder.build(device)
            texts = decoder.decode(emb, topk=1)
            print(json.dumps({"decoded_texts": texts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
