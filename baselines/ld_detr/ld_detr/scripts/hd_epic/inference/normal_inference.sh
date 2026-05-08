dset_name=hd_epic

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data}"

# ========================================
# Configuration Options (can be overridden)
# ========================================
# Checkpoint selection: standard, ours
CKPT_TYPE="${CKPT_TYPE:-standard}"
# Evaluation version for underspecified: s1, s2, s3 (only used if CKPT_TYPE=ours or EVAL_VERSION is explicitly set)
EVAL_VERSION="${EVAL_VERSION:-s1}"

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
    ckpt_path=${DATA_ROOT}/beyond_vmr_models/LD-DETR/hd_epic/results_hd_epic_no_sa_qd_0,25/hd_epic-txt_original/model_best.ckpt
else
    # default to standard
    ckpt_path=${DATA_ROOT}/beyond_vmr_models/LD-DETR/hd_epic/results_hd_epic_standard_ce/hd_epic-txt_original/model_best.ckpt
fi

# ========================================
# Select evaluation path and text features based on EVAL_VERSION
# ========================================
case ${EVAL_VERSION} in
    original)
        eval_path=${DATA_ROOT}/data/hd_epic/annotations/original_annos/HD_EPIC_Narrations_val.jsonl
        t_feat_dir=${DATA_ROOT}/data/hd_epic/features/txt_feats_original/
        ;;
    s1)
        eval_path=${DATA_ROOT}/data/hd_epic/annotations/annotations_s1/HD_EPIC_Narrations_underspecified_full_queries_llm_val_merged_window_500.jsonl
        t_feat_dir=${DATA_ROOT}/data/hd_epic/features/txt_feats_s1/
        ;;
    s2)
        eval_path=${DATA_ROOT}/data/hd_epic/annotations/annotations_s2/HD_EPIC_Narrations_underspecified_full_queries_llm_val_merged_window_500.jsonl
        t_feat_dir=${DATA_ROOT}/data/hd_epic/features/txt_feats_s2/
        ;;
    s3)
        eval_path=${DATA_ROOT}/data/hd_epic/annotations/annotations_s3/HD_EPIC_Narrations_underspecified_full_queries_llm_val_merged_window_500.jsonl
        t_feat_dir=${DATA_ROOT}/data/hd_epic/features/txt_feats_s3/
        ;;
    *)
        echo "Error: Unknown EVAL_VERSION '${EVAL_VERSION}'. Options: original, s1, s2, s3"
        exit 1
        ;;
esac

# Evaluation settings
eval_split_name=val

# Feature directories
feat_root=${DATA_ROOT}/data/hd_epic
v_feat_dirs=(${feat_root}/features/vid_feats)

num_workers=8

# Ensure PYTHONPATH includes the project root
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}:."

echo "Dataset: ${dset_name}"
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
