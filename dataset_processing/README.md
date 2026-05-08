# Dataset Processing Pipeline

This directory contains the complete pipeline for preparing the search-query benchmarks (HD-EPIC-S, ANC-S, YC2-S) from original caption-based datasets.

## Overview

The pipeline consists of 4 main stages:

```
Original Datasets (HD-EPIC, ActivityNet-Captions, YouCook2)
    ↓
1. Feature Extraction (InternVideo2) → Video & Text Features
    ↓
2. Underspecify Queries (Gemma LLM) → Simplified Queries
    ↓
3. Validate & Match Queries → Quality Assurance
    ↓
4. Prepare Final Data (Merge, Split) → Training/Validation Sets
    ↓
Final Datasets: *_merged_window_500.jsonl
```

## Stage 1: Feature Extraction

**Location:** External (InternVideo2)

**What it does:** Extracts visual and textual features from videos and captions

**How to run:**
- Use [InternVideo2](https://github.com/OpenGVLab/InternVideo2) or any other Vision-Language Model (VLM)
- Extract visual features: `feats_dir/vid_feats/`
- Extract text features: `feats_dir/txt_feats_original/` (and variants like `txt_feats_v1` for different query types)

**Output structure:**
```
data/
├── HD_EPIC/
│   ├── internVideo/
│   │   ├── vid_feats/
│   │   ├── txt_feats_original/
│   │   ├── txt_feats_v1/
│   │   ├── txt_feats_v2/
│   │   └── txt_feats_v3/
│   └── annotations/original_annos/
├── activitynet_captions/
│   ├── internVideo/
│   │   ├── vid_feats/
│   │   ├── txt_feats_original/
│   │   └── txt_feats_v1/
│   └── annotations/original_annos/
└── yc2/
    ├── internVideo/
    │   ├── vid_feats/
    │   ├── txt_feats_original/
    │   └── txt_feats_v1/
    └── annotations/original_annos/
```

---

## Stage 2: Underspecify Queries

**Location:** `2_underspecify_queries/`

**What it does:** Transforms caption-based queries into search queries using Gemma LLM

**Requirements:**
- Activate the Gemma environment: `source gemma_env/bin/activate`
- Requires GPU (for LLM inference)

**How to run:**

```bash
# For HD-EPIC
python 2_underspecify_queries/hd_epic/underspecify_queries_hd_epic.py \
  --input_file data/HD_EPIC/annotations/original_annos/HD_EPIC_Narrations_train.jsonl \
  --output_file data/HD_EPIC/annotations/underspecified/original/HD_EPIC_Narrations_train.jsonl \
  --version original  # or s1, s2, s3 for variants

# For ActivityNet-Captions
python 2_underspecify_queries/activitynet_captions/underspecify_queries.py \
  --input_file data/activitynet_captions/annotations/original_annos/train.jsonl \
  --output_file data/activitynet_captions/annotations/underspecified/train.jsonl

# For YouCook2
python 2_underspecify_queries/yc2/underspecify_queries.py \
  --input_file data/yc2/annotations/original_annos/train.jsonl \
  --output_file data/yc2/annotations/underspecified/train.jsonl
```

**Scripts:**
- `hd_epic/underspecify_queries_hd_epic.py` - Main HD-EPIC underspecification (v1, v2, v3, v4 variants available)
- `activitynet_captions/underspecify_queries.py` - ActivityNet-Captions underspecification
- `yc2/underspecify_queries.py` - YouCook2 underspecification

---

## Stage 3: Validate & Match Queries

**Location:** `3_validate_queries/`

**What it does:**
- Validates quality of underspecified queries
- Matches underspecified queries to relevant video moments
- Groups related queries to ensure consistency

**How to run:**

```bash
# Validate and match underspecified queries
python 3_validate_queries/matching_window_based.py \
  --input_file data/HD_EPIC/annotations/underspecified/original/HD_EPIC_Narrations_train.jsonl \
  --output_file data/HD_EPIC/annotations/underspecified/original/HD_EPIC_Narrations_train_matched.jsonl \
  --window_size 500 --fps 3


**Scripts:**
- `matching_window_based.py` - Merges similar queries within temporal windows
- `split_annotations.py` - (Optional) Splits annotations for quality control

---

## Stage 4: Prepare Final Data

**Location:** `4_prepare_data/`

**What it does:**
- Splits data into single-moment vs multi-moment instances
- Generates final training/validation splits
- Creates merged window datasets (`*_merged_window_500.jsonl`)

**How to run:**

```bash
# Create merged window dataset (MAIN OUTPUT)
python 4_prepare_data/merge_windows.py \
  --input_file data/HD_EPIC/annotations/underspecified/original/HD_EPIC_Narrations_train_matched.jsonl \
  --output_file data/HD_EPIC/annotations/underspecified/original/HD_EPIC_Narrations_train_merged_window_500.jsonl \
  --window_size 500 --fps 3

```

**Scripts:**
- `merge_windows.py` - Creates merged window datasets (main output)
- `split_annotations.py` - Creates train/validation splits

---

## Complete Workflow Example

```bash
#!/bin/bash

# Set paths
DATA_ROOT="/data"
DATASET="hd_epic"  # or activitynet_captions, yc2

# Activate Gemma environment for LLM-based underspecification
source gemma_env/bin/activate

# Stage 2: Underspecify queries
python 2_underspecify_queries/${DATASET}/underspecify_queries_${DATASET}.py \
  --input_file ${DATA_ROOT}/${DATASET}/annotations/original_annos/train.jsonl \
  --output_file ${DATA_ROOT}/${DATASET}/annotations/underspecified/train.jsonl

# Stage 3: Validate and match queries
python 3_validate_queries/matching_window_based.py \
  --input_file ${DATA_ROOT}/${DATASET}/annotations/underspecified/train.jsonl \
  --output_file ${DATA_ROOT}/${DATASET}/annotations/underspecified/train_matched.jsonl

# Stage 4: Create merged window dataset
python 4_prepare_data/merge_windows.py \
  --input_file ${DATA_ROOT}/${DATASET}/annotations/underspecified/train_matched.jsonl \
  --output_file ${DATA_ROOT}/${DATASET}/annotations/underspecified/train_merged_window_500.jsonl

# Optional: Split into single/multi-moment
python 4_prepare_data/split_single_multi.py \
  --input_file ${DATA_ROOT}/${DATASET}/annotations/underspecified/train_merged_window_500.jsonl
```