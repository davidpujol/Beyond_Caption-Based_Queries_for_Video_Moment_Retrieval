#!/bin/bash

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data}"

# ========================================
# Set default checkpoint type and evaluation version
# ========================================
CKPT_TYPE="${CKPT_TYPE:-standard}"
EVAL_VERSION="${EVAL_VERSION:-original}"

# ========================================
# Parse command line arguments
# ========================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt-type)
            CKPT_TYPE="$2"
            shift 2
            ;;
        --eval-version)
            EVAL_VERSION="$2"
            shift 2
            ;;
        --ours)
            CKPT_TYPE="ours"
            shift
            ;;
        *)
            break
            ;;
    esac
done

# ========================================
# Select checkpoint based on CKPT_TYPE
# ========================================
case "$CKPT_TYPE" in
    standard)
        ckpt_path=${DATA_ROOT}/beyond_vmr_models/LD-DETR/yc2/results_yc2_standard_ce/yc2-txt_original/model_best.ckpt
        ;;
    ours)
        ckpt_path=${DATA_ROOT}/beyond_vmr_models/LD-DETR/yc2/results_yc2_no_sa_qd_0,25/yc2-txt_original/model_best.ckpt
        ;;
    *)
        echo "Error: Invalid CKPT_TYPE '${CKPT_TYPE}'. Must be 'standard' or 'ours'."
        exit 1
        ;;
esac

# ========================================
# Select evaluation path and text features based on EVAL_VERSION
# ========================================
case "$EVAL_VERSION" in
    original)
        eval_path=${DATA_ROOT}/data/yc2/annotations/original_annos/yc2_reformated_val.jsonl
        t_feat_dir=${DATA_ROOT}/data/yc2/features/txt_feats_original/
        ;;
    underspecified)
        eval_path=${DATA_ROOT}/data/yc2/annotations/annotations_s/val.jsonl
        t_feat_dir=${DATA_ROOT}/data/yc2/features/txt_feats_v1/
        ;;
    *)
        echo "Error: Invalid EVAL_VERSION '${EVAL_VERSION}'. Must be 'original' or 'underspecified'."
        exit 1
        ;;
esac

# Evaluation settings
eval_split_name=val

# Feature directories
feat_root=${DATA_ROOT}/data/yc2
v_feat_dirs=(${feat_root}/features/vid_feats)

num_workers=8

# Ensure PYTHONPATH includes the project root
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}:."

echo "Dataset: yc2"
echo "Checkpoint type: ${CKPT_TYPE}"
echo "Evaluation version: ${EVAL_VERSION}"
echo "Checkpoint: ${ckpt_path}"
echo "Eval path: ${eval_path}"
echo "Text features: ${t_feat_dir}"

# Filter out flags before passing remaining arguments to Python script
filtered_args=()
skip_next=false
for arg in "$@"; do
    if [[ "$skip_next" == true ]]; then
        skip_next=false
        continue
    fi
    case "$arg" in
        --ckpt-type|--eval-version)
            skip_next=true
            continue
            ;;
        --ours)
            continue
            ;;
        *)
            filtered_args+=("$arg")
            ;;
    esac
done

PYTHONPATH=$PYTHONPATH python ld_detr/inference.py \
--resume ${ckpt_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--t_feat_dir ${t_feat_dir} \
--v_feat_dirs ${v_feat_dirs[@]} \
--num_workers ${num_workers} \
--suffix _inference \
"${filtered_args[@]}"
