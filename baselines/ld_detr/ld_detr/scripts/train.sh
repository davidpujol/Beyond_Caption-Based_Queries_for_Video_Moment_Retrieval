#!/bin/bash
# Master training script for LD-DETR with raw InternVideo features
# Usage: bash train.sh <dataset> [--ours] [additional_args]
#
# Datasets: hd_epic, activity_net_captions, yc2
# Example:
#   bash train.sh hd_epic
#   bash train.sh hd_epic --ours
#   bash train.sh yc2 --ours --num_queries 50

set -e

# Check if dataset is provided
if [ $# -lt 1 ]; then
    echo "Usage: bash train.sh <dataset> [--ours] [additional_args]"
    echo ""
    echo "Datasets:"
    echo "  hd_epic                - HD-EPIC dataset"
    echo "  activity_net_captions  - ActivityNet-Captions dataset"
    echo "  yc2                    - YouCook2 dataset"
    echo ""
    echo "Options:"
    echo "  --ours                 - Use the improved model (with architectural modifications)"
    echo ""
    echo "Examples:"
    echo "  bash train.sh hd_epic"
    echo "  bash train.sh hd_epic --ours"
    exit 1
fi

DATASET=$1
USE_OURS=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if --ours flag is set
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --ours)
            USE_OURS=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Validate dataset
case ${DATASET} in
    hd_epic)
        TRAIN_SCRIPT="${SCRIPT_DIR}/hd_epic/train/train$(${USE_OURS} && echo '_ours' || echo '').sh"
        ;;
    activity_net_captions)
        TRAIN_SCRIPT="${SCRIPT_DIR}/activity_net_captions/train/train$(${USE_OURS} && echo '_ours' || echo '').sh"
        ;;
    yc2)
        TRAIN_SCRIPT="${SCRIPT_DIR}/yc2/train/train$(${USE_OURS} && echo '_ours' || echo '').sh"
        ;;
    *)
        echo "Error: Unknown dataset '${DATASET}'"
        echo "Valid datasets: hd_epic, activity_net_captions, yc2"
        exit 1
        ;;
esac

# Check if script exists
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "Error: Training script not found at ${TRAIN_SCRIPT}"
    exit 1
fi

echo "=========================================="
echo "Training LD-DETR on ${DATASET}"
if [ "${USE_OURS}" = true ]; then
    echo "Using improved model (architectural modifications)"
fi
echo "=========================================="
echo ""

# Get the project root and the LD-DETR root
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LDDETR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Set PYTHONPATH to include project root (for 'baselines' module imports)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Change to LD-DETR root directory (where ld_detr module is)
cd "${LDDETR_ROOT}"

# Run the training script with additional arguments
bash "${TRAIN_SCRIPT}" "$@"
