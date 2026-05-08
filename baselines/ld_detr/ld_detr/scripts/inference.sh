#!/bin/bash
# Master inference script for LD-DETR with raw InternVideo features
# Usage: bash inference.sh <dataset> [--ours] [additional_args]
#
# Datasets: hd_epic, activity_net_captions, yc2
# Example:
#   bash inference.sh hd_epic
#   bash inference.sh hd_epic --ours
#   bash inference.sh yc2 --ours --num_workers 8

set -e

# Check if dataset is provided
if [ $# -lt 1 ]; then
    echo "Usage: bash inference.sh <dataset> [--ours] [additional_args]"
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
    echo "  bash inference.sh hd_epic"
    echo "  bash inference.sh hd_epic --ours"
    exit 1
fi

DATASET=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Remove first argument and pass rest to the inference script
shift

# Validate dataset and select inference script
case ${DATASET} in
    hd_epic)
        INFERENCE_SCRIPT="${SCRIPT_DIR}/hd_epic/inference/normal_inference.sh"
        ;;
    activity_net_captions)
        INFERENCE_SCRIPT="${SCRIPT_DIR}/activity_net_captions/inference/normal_inference.sh"
        ;;
    yc2)
        INFERENCE_SCRIPT="${SCRIPT_DIR}/yc2/inference/normal_inference.sh"
        ;;
    *)
        echo "Error: Unknown dataset '${DATASET}'"
        echo "Valid datasets: hd_epic, activity_net_captions, yc2"
        exit 1
        ;;
esac


# Check if script exists
if [ ! -f "${INFERENCE_SCRIPT}" ]; then
    echo "Error: Inference script not found at ${INFERENCE_SCRIPT}"
    exit 1
fi

echo "=========================================="
echo "Running LD-DETR inference on ${DATASET}"
echo "=========================================="
echo ""

# Get the project root and the LD-DETR root
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LDDETR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Set PYTHONPATH to include project root (for 'baselines' module imports)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Change to LD-DETR root directory (where ld_detr module is)
cd "${LDDETR_ROOT}"

# Run the inference script with additional arguments
bash "${INFERENCE_SCRIPT}" "$@"

