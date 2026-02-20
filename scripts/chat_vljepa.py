#!/usr/bin/env python3
from __future__ import annotations

"""Interactive multi-turn chat CLI for VL-JEPA embedding-space inference.

Design note:
- This is a conversational shell around the existing embedding predictor.
- Responses are selected from a fixed text bank using nearest-neighbor decoding.
- It feels "chat-like" in interaction, but it is not an autoregressive LLM.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision.transforms import v2

from vljepa.inference.decoder import NearestNeighborDecoder
from vljepa.train.build import build_model
from vljepa.utils.config import load_yaml, validate_train_config


def parse_args() -> argparse.Namespace:
    """Parse command-line options for interactive VL-JEPA chat."""

    parser = argparse.ArgumentParser(description="Interactive VL-JEPA chat shell")
    parser.add_argument("--config", type=str, default="vljepa/configs/inference_tiny.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text-bank", type=str, required=True, help="txt or jsonl text candidates")
    parser.add_argument("--image", type=str, required=True, help="conversation image context")
    parser.add_argument("--topk", type=int, default=1, help="number of candidate replies to print")
    parser.add_argument("--max-history-turns", type=int, default=6, help="kept user+assistant turns")
    parser.add_argument("--system-prompt", type=str, default="Answer briefly and accurately.")
    return parser.parse_args()


def load_text_bank(path: str | Path) -> List[str]:
    """Load candidate responses from txt or JSONL with `text` key."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Text bank not found: {p}")

    items: List[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("{"):
            row = json.loads(line)
            text = row.get("text", "").strip()
            if text:
                items.append(text)
        else:
            items.append(line)
    if not items:
        raise ValueError("Text bank is empty after parsing.")
    return items


def load_image_frames(image_path: str | Path, num_frames: int, image_size: int) -> torch.Tensor:
    """Load image once and replicate across T frames, matching train/infer conventions."""

    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    if num_frames <= 0:
        raise ValueError("`num_frames` must be > 0.")

    transform = v2.Compose(
        [
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    img = Image.open(p).convert("RGB")
    frame = transform(img)
    return frame.unsqueeze(0).repeat(num_frames, 1, 1, 1)  # [T, C, H, W]


def build_context_query(
    system_prompt: str,
    history: List[Tuple[str, str]],
    new_user_text: str,
    max_history_turns: int,
) -> str:
    """Serialize dialogue into a single predictor query string.

    Paper alignment:
    - VL-JEPA predictor consumes a query text sequence (Sec. 3.1).
    - For multi-turn behavior, we encode short chat history in that same field.
    """

    # Keep first-turn behavior close to training/inference (plain question only),
    # which is more stable for tiny offline checkpoints.
    if not history or max_history_turns == 0:
        return new_user_text.strip()

    capped = history[-(2 * max_history_turns) :]
    lines: List[str] = []
    if system_prompt.strip():
        lines.append(f"Instruction: {system_prompt.strip()}")
    lines.append("Conversation memory:")
    for role, text in capped:
        role_name = "Q" if role == "user" else "A"
        lines.append(f"{role_name}: {text.strip()}")
    lines.append(f"Q: {new_user_text.strip()}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.topk <= 0:
        raise ValueError("`--topk` must be > 0.")
    if args.max_history_turns < 0:
        raise ValueError("`--max-history-turns` must be >= 0.")

    cfg = load_yaml(args.config)
    validate_train_config(cfg, require_runtime=True, require_train=False)
    device = torch.device(cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu")

    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing `model` state dict: {args.checkpoint}")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()

    text_bank = load_text_bank(args.text_bank)
    decoder = NearestNeighborDecoder(model, text_bank)
    decoder.build(device)

    frames = load_image_frames(
        image_path=args.image,
        num_frames=cfg["data"]["num_frames"],
        image_size=cfg["data"]["image_size"],
    ).unsqueeze(0).to(device)  # [1, T, C, H, W]

    history: List[Tuple[str, str]] = []
    print("Interactive VL-JEPA chat. Commands: /exit, /reset, /image <path>, /help")
    print(f"Context image: {args.image}")

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_text:
            continue
        if user_text == "/exit":
            print("Exiting.")
            break
        if user_text == "/reset":
            history.clear()
            print("assistant> Conversation history cleared.")
            continue
        if user_text == "/help":
            print("assistant> Commands: /exit, /reset, /image <path>, /help")
            continue
        if user_text.startswith("/image "):
            new_path = user_text.split(" ", 1)[1].strip()
            frames = load_image_frames(
                image_path=new_path,
                num_frames=cfg["data"]["num_frames"],
                image_size=cfg["data"]["image_size"],
            ).unsqueeze(0).to(device)
            history.clear()
            print(f"assistant> Context image updated: {new_path}")
            print("assistant> Conversation history cleared for the new image.")
            continue

        query = build_context_query(
            system_prompt=args.system_prompt,
            history=history,
            new_user_text=user_text,
            max_history_turns=args.max_history_turns,
        )
        with torch.no_grad():
            pred = model.predict_embedding(frames, [query])
            results = decoder.decode(pred, topk=args.topk)

        # Decoder returns List[str] for topk=1 or List[List[str]] for topk>1.
        if args.topk == 1:
            reply = results[0]
            print(f"assistant> {reply}")
        else:
            choices = results[0]
            reply = choices[0]
            print("assistant> Top candidates:")
            for i, cand in enumerate(choices, start=1):
                print(f"  {i}. {cand}")

        history.append(("user", user_text))
        history.append(("assistant", reply))


if __name__ == "__main__":
    main()
