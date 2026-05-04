# VL-JEPA Joint Embedding Predictive Architecture for Vision-language (Paper-Derived Implementation)

This repository contains a full implementation of the architecture described in:
`VL-JEPA: Joint Embedding Predictive Architecture for Vision-language` (arXiv:2512.10942v2, Feb 2, 2026).

The code includes:
- VL-JEPA model (`X-Encoder`, query-conditioned `Predictor`, `Y-Encoder`)
- Bi-directional InfoNCE training objective in embedding space
- Two-stage training pipelines (`pretraining`, `SFT`)
- Inference for captioning, discriminative VQA, and selective decoding
- Retrieval/classification utilities via embedding similarity
- Unit tests for critical utility and algorithmic components

## 1. Complete Setup (CPU and GPU)

### 1.1 Prerequisites

- OS: Linux/macOS (Windows should work with equivalent commands)
- Python: 3.10+ recommended
- Disk: at least 10 GB free for CPU setup, more for GPU + large checkpoints
- Optional for paper-scale configs: internet access to download Hugging Face models/checkpoints
### 1.5 Verify Installation

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import torchvision, PIL, yaml, tqdm; print('deps ok')"
```

Expected:
- CPU setup: `cuda False`
- GPU setup: `cuda True` (if driver/runtime are correct)

### 1.6 Configuration: CPU vs GPU

This repo has two practical config tracks:

- Tiny/offline configs (works without external model downloads):
  - `vljepa/configs/pretrain_tiny.yaml`
  - `vljepa/configs/inference_tiny.yaml`
- Paper-style configs (require heavier backbones/model downloads):
  - `vljepa/configs/pretrain.yaml`
  - `vljepa/configs/sft.yaml`
  - `vljepa/configs/inference.yaml`

Edit these keys in YAML:

- `runtime.device`: `cpu` or `cuda`
- `model.vision_backbone`:
  - tiny/offline: `toy_cnn`
  - heavier: `vit_b_16`, `vit_l_16`, or supported HF model via `hf:<model_name>`
- `model.query_model_name` and `model.y_encoder_name`:
  - tiny/offline: `toy`
  - paper-style: real Hugging Face model names

### 1.7 Quick Start Commands

CPU tiny end-to-end:

```bash
PYTHONPATH=. python scripts/make_tiny_data.py
PYTHONPATH=. python scripts/train_pretrain.py --config vljepa/configs/pretrain_tiny.yaml
PYTHONPATH=. python scripts/run_inference.py \
  --config vljepa/configs/inference_tiny.yaml \
  --checkpoint outputs/tiny_pretrain/step_0000002.pt \
  --manifest data/tiny_infer_manifest.jsonl \
  --text-bank data/tiny_text_bank.txt \
  --mode caption
```

Interactive chat loop:

```bash
PYTHONPATH=. python scripts/chat_vljepa.py \
  --config vljepa/configs/inference_tiny.yaml \
  --checkpoint outputs/tiny_pretrain/step_0000002.pt \
  --text-bank data/tiny_text_bank.txt \
  --image data/tiny/red.png
```

### 1.8 Troubleshooting

- `ModuleNotFoundError: vljepa`
  - Run commands with `PYTHONPATH=.` from repository root.
- CUDA not detected even on GPU machine
  - Check `nvidia-smi`, CUDA driver compatibility, and PyTorch CUDA wheel index.
- Out-of-memory on GPU
  - Reduce batch size, image size, num frames, or use tiny configs first.
- Download/auth failures for HF models
  - Use tiny configs (`toy_*`) for offline validation first.

## 2. Data Format

Training uses JSONL manifests. Each line:

```json
{"image":"path/to/image.jpg","query":"Describe the scene.","target":"A person is cooking."}
```

or:

```json
{"video":"path/to/video.mp4","query":"What happens next?","target":"The person opens the door.","candidates":["open door","sit down","wash hands"]}
```

Examples:
- `data/pretrain_manifest.example.jsonl`
- `data/sft_manifest.example.jsonl`

## 3. Paper Alignment

Implemented paper choices:
- Frozen visual encoder + trainable predictor + trainable text target encoder.
- Shared projection space with 1536-d target embedding space.
- Query-conditioned embedding prediction.
- Bi-directional InfoNCE for anti-collapse and alignment.
- Y-encoder learning-rate multiplier (`0.05` default).
- Two-stage schedule structure:
  - Stage 1 pretraining: constant LR (`5e-5`)
  - Stage 2 SFT: cosine annealing
- Selective decoding using temporal segmentation with Ward-style agglomerative clustering.

Notes:
- The paper uses V-JEPA2 and specific internal data mixtures (Datacomp/YFCC/Action100M/PLM). This implementation provides the same training logic and interfaces, with user-provided datasets/checkpoints.
- Default backbone/model names are placeholders that can be replaced with available checkpoints.

## 4. Train

### 4.1 Pretraining stage

1. Copy template manifest:
```bash
cp data/pretrain_manifest.example.jsonl data/pretrain_manifest.jsonl
```

2. Run:
```bash
python scripts/train_pretrain.py --config vljepa/configs/pretrain.yaml
```

### 4.2 SFT stage

1. Copy template manifest:
```bash
cp data/sft_manifest.example.jsonl data/sft_manifest.jsonl
```

2. Run:
```bash
python scripts/train_sft.py \
  --config vljepa/configs/sft.yaml \
  --checkpoint outputs/pretrain/step_0001000.pt
```

## 5. Inference

### 5.1 Captioning (embedding -> text via lightweight NN decoder)

```bash
python scripts/run_inference.py \
  --config vljepa/configs/inference.yaml \
  --checkpoint outputs/sft/step_0001000.pt \
  --manifest data/sft_manifest.jsonl \
  --text-bank data/text_bank.example.txt \
  --mode caption
```

### 5.2 Discriminative VQA

```bash
python scripts/run_inference.py \
  --config vljepa/configs/inference.yaml \
  --checkpoint outputs/sft/step_0001000.pt \
  --manifest data/sft_manifest.jsonl \
  --mode discriminative_vqa
```

### 5.3 Selective decoding on embedding stream

```bash
python scripts/run_inference.py \
  --config vljepa/configs/inference.yaml \
  --checkpoint outputs/sft/step_0001000.pt \
  --manifest data/single_video.jsonl \
  --text-bank data/text_bank.example.txt \
  --mode selective \
  --num-segments 12
```

## 6. Offline Tiny Smoke Test

Use this path to verify end-to-end execution without external model downloads:
- local toy vision/text encoders
- synthetic local image data
- CPU-only tiny run

Commands:

```bash
python scripts/make_tiny_data.py
python scripts/train_pretrain.py --config vljepa/configs/pretrain_tiny.yaml
python scripts/run_inference.py \
  --config vljepa/configs/inference_tiny.yaml \
  --checkpoint outputs/tiny_pretrain/step_0000002.pt \
  --manifest data/tiny_infer_manifest.jsonl \
  --text-bank data/tiny_text_bank.txt \
  --mode caption
```

Expected smoke-test behavior:
- training finishes 2 steps and saves checkpoints under `outputs/tiny_pretrain/`
- inference prints one JSON line with a predicted label from the tiny text bank

## 7. Interactive Chat Loop

You can talk to the model in a multi-turn terminal loop:

```bash
python scripts/chat_vljepa.py \
  --config vljepa/configs/inference_tiny.yaml \
  --checkpoint outputs/tiny_pretrain/step_0000002.pt \
  --text-bank data/tiny_text_bank.txt \
  --image data/tiny/red.png
```

Inside the prompt:
- type normal text to ask questions
- `/image path/to/new_image.png` switches visual context
- `/reset` clears conversation memory
- `/exit` quits

Important:
- this is chat-style interaction around VL-JEPA embedding retrieval
- replies are selected from the provided text bank (not free-form token generation)

## 8. Repository Layout

- `vljepa/models/vljepa.py`: core VL-JEPA model and parameter groups
- `vljepa/models/losses.py`: bi-directional InfoNCE
- `vljepa/train/trainer.py`: training loop/checkpointing
- `vljepa/inference/decoder.py`: lightweight readout decoder
- `vljepa/inference/selective.py`: temporal selective decoding
- `vljepa/eval/tasks.py`: discriminative match/retrieval helpers
- `scripts/*.py`: CLI entrypoints (`train`, `inference`, `chat`)

## 9. Reproducibility And Traceability

- Determinism:
  - training scripts call `set_seed(..., deterministic=True)`
  - deterministic algorithm mode is enabled where possible
- Explicit config control:
  - all hyperparameters live in YAML config files
  - each run writes `resolved_config.yaml` into its output directory
- Run metadata:
  - trainer writes `run_meta.json` (PyTorch/CUDA/device/platform)
- Logs and checkpoints:
  - step-wise JSONL logs: `train_log.jsonl`
- periodic checkpoints: `step_XXXXXXX.pt`

## 10. Tests

Run:

```bash
pytest -q
```

Current tests cover:
- InfoNCE loss correctness/guards
- selective decoding segmentation behavior
- config validation and RNG determinism

## 11. Reference

- Delong Chen, Mustafa Shukor, Théo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla,
  Yejin Bang, Allen Bolourchi, Yann LeCun, Pascale Fung.
  `VL-JEPA: Joint Embedding Predictive Architecture for Vision-language`.
  arXiv:2512.10942v2, February 2, 2026.
