#!/bin/bash

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data}"

# ========================================
# Configuration Options (can be overridden)
# ========================================
# Checkpoint selection: standard, ours
CKPT_TYPE="${CKPT_TYPE:-standard}"
# Evaluation version: original, underspecified (default: original)
EVAL_VERSION="${EVAL_VERSION:-original}"

# Parse command-line flags
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
if [[ "$CKPT_TYPE" == "ours" ]]; then
    ckpt_path=${DATA_ROOT}/beyond_vmr_models/LD-DETR/anet/results_no_sa_qd_0,25_bsz_16_lr_1e-4/hl-txt_original/model_best.ckpt
else
    # default to standard
    ckpt_path=${DATA_ROOT}/beyond_vmr_models/LD-DETR/anet/results_standard_ce_bsz_16_lr_1e-4/hl-txt_original/model_best.ckpt
fi

# ========================================
# Select evaluation path and text features based on EVAL_VERSION
# ========================================
case ${EVAL_VERSION} in
    original)
        eval_path=${DATA_ROOT}/data/anet/annotations/original_annos/val.jsonl
        t_feat_dir=${DATA_ROOT}/data/anet/features/txt_feats_original/
        ;;
    underspecified)
        eval_path=${DATA_ROOT}/data/anet/annotations/annotations_s/val.jsonl
        t_feat_dir=${DATA_ROOT}/data/anet/features/txt_feats_underspecified_v3/
        ;;
    *)
        echo "Error: Unknown EVAL_VERSION '${EVAL_VERSION}'. Options: original, underspecified"
        exit 1
        ;;
esac

# Evaluation settings
eval_split_name=val

# Feature directories
feat_root=${DATA_ROOT}/data/anet
v_feat_dirs=(${feat_root}/features/vid_feats)

num_workers=8

# Ensure PYTHONPATH includes the project root
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}:."

echo "Dataset: hl"
echo "Checkpoint Type: ${CKPT_TYPE}"
echo "Evaluation Version: ${EVAL_VERSION}"
echo "Checkpoint: ${ckpt_path}"
echo "Eval path: ${eval_path}"
echo "Text features: ${t_feat_dir}"

PYTHONPATH=$PYTHONPATH python ld_detr/inference.py \
--resume ${ckpt_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--t_feat_dir ${t_feat_dir} \
--v_feat_dirs ${v_feat_dirs[@]} \
--num_workers ${num_workers} \
--suffix _inference
# Filter out custom flags before passing remaining args to Python script
args=()
for arg in "$@"; do
    case $arg in
        --ckpt-type|--eval-version|--ours)
            # Skip these and their potential values
            if [[ "$arg" == "--ckpt-type" ]] || [[ "$arg" == "--eval-version" ]]; then
                shift  # Skip the flag itself; its value was already processed
            fi
            ;;
        *)
            args+=("$arg")
            ;;
    esac
done
${args[@]}
