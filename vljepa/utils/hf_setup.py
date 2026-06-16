from __future__ import annotations

import os
from pathlib import Path


def setup_hf(env_file: str | Path = ".env/hf.env") -> None:
    """Load HF token, redirect HF cache to ./ckpts, login."""
    env_path = Path(env_file)
    if not env_path.exists():
        raise FileNotFoundError(f"HF env file not found: {env_path}")

    token: str | None = None
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip()
                break

    if not token:
        raise ValueError(f"HF_TOKEN not found in {env_path}")

    project_root = Path(__file__).resolve().parents[2]
    ckpts_dir = project_root / "ckpts"
    ckpts_dir.mkdir(exist_ok=True)
    os.environ["HF_HOME"] = str(ckpts_dir)

    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
