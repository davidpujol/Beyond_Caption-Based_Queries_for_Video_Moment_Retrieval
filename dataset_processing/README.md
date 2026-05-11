# Dataset Processing Pipeline

This directory contains the complete pipeline for preparing the search-query benchmarks (HD-EPIC-S, ANC-S, YC2-S) from original caption-based datasets.

## Overview

The pipeline consists of 3 main stages (**Feature Extraction is performed LAST**):

```
Original Datasets (HD-EPIC, ActivityNet-Captions, YouCook2)
    ↓
[STAGE 1] Underspecify Queries (Gemma LLM)
    ↓ Simplified queries in *_underspecified.jsonl format
    ↓
[STAGE 2] Match & Prepare Queries (Temporal Windowing)
    ↓ Matched queries in *_merged_window_500.jsonl format
    ↓
[STAGE 3] Feature Extraction (InternVideo2) ← PERFORMED LAST
    ↓ Extract video & text features for matched queries only
    ↓
Final Output: *_merged_window_500.jsonl + features for training
```

---

## Stage 1: Underspecify Queries (Gemma LLM)

**Location:** `2_underspecify_queries/`

**What it does:**
- Transforms caption-based queries into search queries using Gemma LLM
- Simplifies descriptive captions into short, concise search queries
- Supports multiple simplification levels (v1, v2, v3, v4 for HD-EPIC)

**Prerequisites:**
- Activate the Gemma environment: `source gemma_env/bin/activate`
- Requires GPU for LLM inference
- Original annotation files in `${DATA_ROOT}/{dataset}/annotations/original_annos/`

**How to run:**

```bash
# Set environment
export DATA_ROOT=/path/to/data
source gemma_env/bin/activate

# For HD-EPIC (multiple variants available)
python 2_underspecify_queries/hd_epic/underspecify_queries_hd_epic.py \
  --input_file ${DATA_ROOT}/HD_EPIC/annotations/original_annos/train.jsonl \
  --output_file ${DATA_ROOT}/HD_EPIC/annotations/underspecified/original/train.jsonl \
  --version original

# For HD-EPIC variants (s1, s2, s3)
python 2_underspecify_queries/hd_epic/underspecify_queries_hd_epic_v1.py \
  --input_file ${DATA_ROOT}/HD_EPIC/annotations/original_annos/train.jsonl \
  --output_file ${DATA_ROOT}/HD_EPIC/annotations/underspecified/s1/train.jsonl

# For ActivityNet-Captions
python 2_underspecify_queries/activitynet_captions/underspecify_queries.py \
  --input_file ${DATA_ROOT}/activitynet_captions/annotations/original_annos/train.jsonl \
  --output_file ${DATA_ROOT}/activitynet_captions/annotations/underspecified/train.jsonl

# For YouCook2
python 2_underspecify_queries/yc2/underspecify_queries.py \
  --input_file ${DATA_ROOT}/yc2/annotations/original_annos/train.jsonl \
  --output_file ${DATA_ROOT}/yc2/annotations/underspecified/train.jsonl
```

**Scripts:**
- `hd_epic/underspecify_queries_hd_epic.py` - Main HD-EPIC underspecification
- `hd_epic/underspecify_queries_hd_epic_v1.py` - HD-EPIC variant 1 (s1)
- `hd_epic/underspecify_queries_hd_epic_v2.py` - HD-EPIC variant 2 (s2)
- `hd_epic/underspecify_queries_hd_epic_v3.py` - HD-EPIC variant 3 (s3)
- `hd_epic/underspecify_queries_hd_epic_v4.py` - HD-EPIC variant 4
- `activitynet_captions/underspecify_queries.py` - ActivityNet underspecification
- `activitynet_captions/underspecify_queries3.py` - ActivityNet alternative variant
- `yc2/underspecify_queries.py` - YouCook2 underspecification

---

## Stage 2: Match & Prepare Queries

**Location:** `3_matching_and_preparation/`

**What it does:**
This stage has 3 substeps:
1. **Match similar queries:** Merge multiple captions that represent the same underspecified query using semantic similarity
2. **Generate window-wise annotations:** Accumulate all relevant windows for each query that fall within the same temporal window (500ms)
3. **Split into train/validation:** Create final train/validation splits for each dataset

**How to run:**

```bash
export DATA_ROOT=/path/to/data

# ============================================
# Step 1: Merge similar queries for each dataset
# ============================================

# For HD-EPIC (all variants)
python 3_matching_and_preparation/matching_queries_hd.py \
  '${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v1/HD_EPIC_Narrations_underspecified_full_queries_llm.jsonl' \
  '${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v1/HD_EPIC_Narrations_underspecified_merged.jsonl' \
  --threshold 0.85

python 3_matching_and_preparation/matching_queries_hd.py \
  '${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v2/HD_EPIC_Narrations_underspecified_full_queries_llm.jsonl' \
  '${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v2/HD_EPIC_Narrations_underspecified_merged.jsonl' \
  --threshold 0.85

# For ActivityNet-Captions
python 3_matching_and_preparation/matching_queries_anet.py \
  '${DATA_ROOT}/activitynet_captions/underspecified/HD_EPIC_Narrations_underspecified_full_queries_llm.jsonl' \
  '${DATA_ROOT}/activitynet_captions/underspecified/HD_EPIC_Narrations_underspecified_merged.jsonl' \
  --threshold 0.85

# For YouCook2
python 3_matching_and_preparation/matching_queries_yc2.py \
  '${DATA_ROOT}/yc2/underspecified/HD_EPIC_Narrations_underspecified_full_queries_llm.jsonl' \
  '${DATA_ROOT}/yc2/underspecified/HD_EPIC_Narrations_underspecified_merged.jsonl' \
  --threshold 0.85

# ============================================
# Step 2: Generate window-wise annotations
# ============================================

python 3_matching_and_preparation/matching_window_based.py \
  '${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v1/HD_EPIC_Narrations_underspecified_merged.jsonl' \
  --window_size 500 --fps 3

python 3_matching_and_preparation/matching_window_based.py \
  '${DATA_ROOT}/activitynet_captions/underspecified/HD_EPIC_Narrations_underspecified_merged.jsonl' \
  --window_size 500 --fps 3

python 3_matching_and_preparation/matching_window_based.py \
  '${DATA_ROOT}/yc2/underspecified/HD_EPIC_Narrations_underspecified_merged.jsonl' \
  --window_size 500 --fps 3

# ============================================
# Step 3: Split into train/validation
# ============================================

# For HD-EPIC (split_dir contains train_videos.txt and val_videos.txt)
python 3_matching_and_preparation/split_annotations_hd_epic.py \
  '${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v1/HD_EPIC_Narrations_underspecified_merged_window_500.jsonl' \
  '${DATA_ROOT}/HD_EPIC/splits' \
  '${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v1/'

# For ActivityNet-Captions & YouCook2
python 3_matching_and_preparation/split_annotations_anet_yc2.py \
  '${DATA_ROOT}/activitynet_captions/underspecified/HD_EPIC_Narrations_underspecified_merged_window_500.jsonl' \
  '${DATA_ROOT}/activitynet_captions/underspecified/'

python 3_matching_and_preparation/split_annotations_anet_yc2.py \
  '${DATA_ROOT}/yc2/underspecified/HD_EPIC_Narrations_underspecified_merged_window_500.jsonl' \
  '${DATA_ROOT}/yc2/underspecified/'
```

---

## Stage 3: Feature Extraction (InternVideo2)

**Location:** External (perform LAST, after Stage 1-2)

**What it does:**
- Extracts visual features from videos using InternVideo2
- Extracts text features for underspecified queries
- **Only extract features for queries that passed Stage 2 (matched queries)**

**How to run:**

Use [InternVideo2](https://github.com/OpenGVLab/InternVideo2) or any Vision-Language Model (VLM):

**Files needed for training:**
- `*_merged_window_500.jsonl` - Annotation files from Stage 2
- `internVideo/vid_feats/` - Video features
- `internVideo/txt_feats_v*` - Text features (variant-specific)
