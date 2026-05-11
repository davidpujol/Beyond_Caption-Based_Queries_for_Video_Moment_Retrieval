# Beyond Caption-Based Queries for Video Moment Retrieval

Official repository for the paper **"Beyond Caption-Based Queries for Video Moment Retrieval"**

## Abstract

Current Video Moment Retrieval (VMR) models are trained on videos paired with captions, which are written by annotators after watching the videos. These captions are used as textual queries—which we term *caption-based queries*. This annotation process induces a visual bias, leading to overly descriptive and fine-grained queries, which significantly differ from the more general search queries that users are likely to employ in practice.

In this work, we investigate the degradation of existing VMR methods, particularly DETR architectures, when trained on caption-based queries but evaluated on search queries. We introduce three benchmarks by modifying the textual queries in three public VMR datasets: HD-EPIC, YouCook2, and ActivityNet-Captions.

Our analysis reveals two key generalization challenges:
- **(i) Language gap:** Arising from the linguistic under-specification of search queries
- **(ii) Multi-moment gap:** Caused by the shift from single-moment to multi-moment queries

We identify a critical issue in these architectures—an *active decoder-query collapse*—as a primary cause of poor generalization to multi-moment instances. We mitigate this issue with architectural modifications that effectively increase the number of active decoder queries.

**Key Results:** Our approach improves performance on search queries by up to **14.82% mAPm**, and up to **21.83% mAPm** on multi-moment search queries.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data](#data)
- [Training](#training)
- [Inference](#inference)
- [Privacy & Sanitization](#privacy--sanitization)
- [Citation](#citation)

## Prerequisites

- Python 3.9+
- CUDA 11.8+
- GPU with sufficient VRAM (≥24GB recommended)
- Docker (optional, for containerized setup)

## Data

### Data Pipeline Overview

The complete pipeline for preparing the search-query benchmarks:

```
Original Datasets (HD-EPIC, ActivityNet-Captions, YouCook2)
    ↓
[STAGE 1] Underspecify Queries (Gemma LLM)
    ↓ Simplified queries
    ↓
[STAGE 2] Match & Prepare Queries
    ├─ Step 1: Merge similar queries (semantic similarity)
    ├─ Step 2: Generate window-wise annotations (temporal windows)
    └─ Step 3: Split into train/validation
    ↓ Final annotations (*_merged_window_500_train/val.jsonl)
    ↓
[STAGE 3] Feature Extraction (InternVideo2) ← PERFORMED LAST
    ↓ Extract video & text features for matched queries only
    ↓
Final Dataset Ready for Training
```

**Quick Start:**
```bash
# Complete pipeline guide:
cd dataset_processing/
cat README.md  # Full step-by-step guide with all commands
```

**Documentation:**
- `dataset_processing/README.md` - Complete pipeline documentation with all 3 stages
  - Stage 1: Underspecify Queries (Gemma LLM)
  - Stage 2: Match & Prepare Queries (merging, windowing, splitting)
  - Stage 3: Feature Extraction (InternVideo2)

### Download Pre-Processed Data

We provide pre-processed datasets (features + underspecified annotations) and model checkpoints for quick start:

#### Option 1: Manual Download (Browser)
- **All-in-One Package** (Model Checkpoints + Features + Annotations): [Google Drive Link](https://drive.google.com/file/d/12oHlqedwwrb45FW8QFt_pWc-dNXewGSS/view?usp=sharing)

After downloading, extract to the repository root:
```bash
unzip downloaded_file.zip -d ./
```

This will create:
- `model_checkpoints/` - Pre-trained model weights
- `data/` - Pre-processed datasets with features and annotations

#### Option 2: Command-Line Download

First, install `gdown`:
```bash
pip install gdown
```

Download all-in-one package:
```bash
gdown 12oHlqedwwrb45FW8QFt_pWc-dNXewGSS -O data_and_models.zip
```

Extract the downloaded files:
```bash
unzip data_and_models.zip -d ./
```

### Datasets

We evaluate on three search-query benchmarks:
- **HD-EPIC-S:** HD-EPIC dataset with search queries (4 variants: original, s1, s2, s3)
- **ANC-S:** ActivityNet-Captions with search queries
- **YC2-S:** YouCook2 with search queries

#### Expected Directory Structure After Data Download
```
./data/
├── HD_EPIC/
│   ├── internVideo/
│   │   ├── vid_feats/
│   │   ├── txt_feats_original/
│   │   ├── txt_feats_v1/
│   │   ├── txt_feats_v2/
│   │   └── txt_feats_v3/
│   └── annotations/
│       ├── original_annos/
│       └── underspecified/
│           ├── original/
│           │   ├── *_merged_window_500.jsonl
│           │   └── ...
│           ├── s1/
│           ├── s2/
│           └── s3/
├── activitynet_captions/
│   ├── internVideo/
│   │   ├── vid_feats/
│   │   ├── txt_feats_original/
│   │   └── txt_feats_v1/
│   └── annotations/
│       └── underspecified/
│           └── *_merged_window_500.jsonl
└── yc2/
    ├── internVideo/
    │   ├── vid_feats/
    │   ├── txt_feats_original/
    │   └── txt_feats_v1/
    └── annotations/
        └── underspecified/
            └── *_merged_window_500.jsonl
```

**Files with pattern `*_merged_window_500.jsonl` are the primary datasets used for training and evaluation.**


## Installation

### Option 1: Docker Setup (Recommended)

Build the Docker image:
```bash
docker build -t beyond_caption_based_queries_vmr:latest .
```

Run the container with GPU support:
```bash
docker run --gpus all -it --rm \
  --name beyond_vmr \
  --shm-size 200gb \
  -v /path/to/project:/workspace \
  -v /path/to/data:/data \
  beyond_caption_based_queries_vmr
```

The Docker image automatically installs all dependencies including:
- PyTorch with CUDA support
- All packages from `environment_cgdetr.yml`
- CG-DETR and LD-DETR requirements

### Option 2: Conda/Pip Setup

Create the conda environment:
```bash
conda env create -f environment_cgdetr.yml
conda activate base
```

Install additional dependencies:
```bash
pip install -r baselines/CGDETR/requirements.txt
# OR
pip install -r baselines/ld_detr/requirements.txt
```

For LLM-based query generation:
```bash
pip install -r requirements_gemma_generation.txt
```


## Training

### Training CG-DETR

**Baseline Model:**
```bash
CUDA_VISIBLE_DEVICES=0 bash baselines/CGDETR/cg_detr/scripts/train.sh hd_epic
CUDA_VISIBLE_DEVICES=0 bash baselines/CGDETR/cg_detr/scripts/train.sh activity_net_captions
CUDA_VISIBLE_DEVICES=0 bash baselines/CGDETR/cg_detr/scripts/train.sh yc2
```

**Improved Model (with architectural modifications):**
```bash
CUDA_VISIBLE_DEVICES=0 bash baselines/CGDETR/cg_detr/scripts/train.sh hd_epic --ours
CUDA_VISIBLE_DEVICES=0 bash baselines/CGDETR/cg_detr/scripts/train.sh activity_net_captions --ours
CUDA_VISIBLE_DEVICES=0 bash baselines/CGDETR/cg_detr/scripts/train.sh yc2 --ours
```

### Training LD-DETR

**Baseline Model:**
```bash
CUDA_VISIBLE_DEVICES=0 bash baselines/ld_detr/ld_detr/scripts/train.sh hd_epic
CUDA_VISIBLE_DEVICES=0 bash baselines/ld_detr/ld_detr/scripts/train.sh activity_net_captions
CUDA_VISIBLE_DEVICES=0 bash baselines/ld_detr/ld_detr/scripts/train.sh yc2
```

**Improved Model (with architectural modifications):**
```bash
CUDA_VISIBLE_DEVICES=0 bash baselines/ld_detr/ld_detr/scripts/train.sh hd_epic --ours
CUDA_VISIBLE_DEVICES=0 bash baselines/ld_detr/ld_detr/scripts/train.sh activity_net_captions --ours
CUDA_VISIBLE_DEVICES=0 bash baselines/ld_detr/ld_detr/scripts/train.sh yc2 --ours
```

### Custom Training Parameters

You can pass additional arguments to customize training:
```bash
CUDA_VISIBLE_DEVICES=0 bash baselines/CGDETR/cg_detr/scripts/train.sh hd_epic --num_queries 50 --n_epoch 200 --bsz 16
```

For dataset-specific configurations, edit the training scripts in `baselines/{CGDETR,ld_detr}/{model}/scripts/{dataset}/train/train.sh`.

### Data Setup

Extract the downloaded data to `./data/`:
```bash
unzip features_annotations.zip -d ./data
```

The scripts expect the following structure:
```
./data/
├── features_annotations/
│   ├── HD_EPIC/
│   ├── activitynet_captions/
│   └── youcook2/
└── results/  # Auto-created during training
```

Override the data location with `DATA_ROOT`:
```bash
DATA_ROOT=/path/to/data CUDA_VISIBLE_DEVICES=0 bash baselines/CGDETR/cg_detr/scripts/train.sh hd_epic
```

## Inference

### Running CG-DETR Inference

**Baseline Model (original annotations):**
```bash
CUDA_VISIBLE_DEVICES=0 bash baselines/CGDETR/cg_detr/scripts/inference.sh hd_epic
CUDA_VISIBLE_DEVICES=0 bash baselines/CGDETR/cg_detr/scripts/inference.sh activity_net_captions
CUDA_VISIBLE_DEVICES=0 bash baselines/CGDETR/cg_detr/scripts/inference.sh yc2
```

**Improved Model (with --ours flag):**
```bash
CUDA_VISIBLE_DEVICES=0 bash baselines/CGDETR/cg_detr/scripts/inference.sh hd_epic --ours
CUDA_VISIBLE_DEVICES=0 bash baselines/CGDETR/cg_detr/scripts/inference.sh activity_net_captions --ours
CUDA_VISIBLE_DEVICES=0 bash baselines/CGDETR/cg_detr/scripts/inference.sh yc2 --ours
```

### Running LD-DETR Inference

**Baseline Model (original annotations):**
```bash
CUDA_VISIBLE_DEVICES=0 bash baselines/ld_detr/ld_detr/scripts/inference.sh hd_epic
CUDA_VISIBLE_DEVICES=0 bash baselines/ld_detr/ld_detr/scripts/inference.sh activity_net_captions
CUDA_VISIBLE_DEVICES=0 bash baselines/ld_detr/ld_detr/scripts/inference.sh yc2
```

**Improved Model (with --ours flag):**
```bash
CUDA_VISIBLE_DEVICES=0 bash baselines/ld_detr/ld_detr/scripts/inference.sh hd_epic --ours
CUDA_VISIBLE_DEVICES=0 bash baselines/ld_detr/ld_detr/scripts/inference.sh activity_net_captions --ours
CUDA_VISIBLE_DEVICES=0 bash baselines/ld_detr/ld_detr/scripts/inference.sh yc2 --ours
```

### Inference Flags

Inference scripts support flexible flags to control checkpoint selection and evaluation configuration:

#### `--ours`
- Automatically sets checkpoint type to improved model and evaluation version to underspecified annotations
- Shorthand equivalent to: `--ckpt-type ours --eval-version underspecified` (for HD-EPIC use `--eval-version s1` or `s2` or `s3`)
- Default: Disabled (uses standard baseline with original annotations)

#### `--ckpt-type [standard|ours]`
- **standard**: Uses the baseline model checkpoint
- **ours**: Uses the improved model checkpoint with architectural modifications
- Default: `standard`

#### `--eval-version [VERSION]`
- For **HD-EPIC**: `original`, `s1`, `s2`, `s3`
- For **ActivityNet-Captions** and **YouCook2**: `original`, `underspecified`
- Default: `original`

#### Examples

Improved model on original annotations:
```bash
CUDA_VISIBLE_DEVICES=0 bash baselines/CGDETR/cg_detr/scripts/inference.sh hd_epic --ckpt-type ours --eval-version original
```

Baseline model on underspecified annotations:
```bash
CUDA_VISIBLE_DEVICES=0 bash baselines/CGDETR/cg_detr/scripts/inference.sh hd_epic --ckpt-type standard --eval-version underspecified
```

Improved model on HD-EPIC s2 variant:
```bash
CUDA_VISIBLE_DEVICES=0 bash baselines/CGDETR/cg_detr/scripts/inference.sh hd_epic --ckpt-type ours --eval-version s2
```

Environment variable alternative:
```bash
CKPT_TYPE=ours EVAL_VERSION=s1 CUDA_VISIBLE_DEVICES=0 bash baselines/CGDETR/cg_detr/scripts/inference.sh hd_epic
```

### Custom Data Location

Override the data location with `DATA_ROOT`:
```bash
DATA_ROOT=/path/to/data bash baselines/CGDETR/cg_detr/scripts/inference.sh hd_epic --ours
```

### Evaluation

After running inference, evaluate the results:
```bash
python baselines/CGDETR/standalone_eval/eval.py \
  --predictions predictions.json \
  --ground_truth annotations.json
```

## Citation

If you use this work, please cite:
```
@InProceedings{pujol2026beyond, author = {Pujol-Perich, David and Clapés, Albert and Damen, Dima and Escalera, Sergio and Wray, Michael}, title = {Beyond Caption-Based Queries for Video Moment Retrieval}, booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, year = {2026}, month = {June} }
```