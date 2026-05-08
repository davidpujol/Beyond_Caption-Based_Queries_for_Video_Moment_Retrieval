dset_name=yc2
ctx_mode=video_tef
v_feat_types=internvideo
t_feat_type=internvideo 

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data}"

results_root=${DATA_ROOT}/results/ld_detr/yc2_debug
exp_id=exp
exp_name=txt_original

######## data paths
train_path=${DATA_ROOT}/data/yc2/annotations/original_annos/yc2_reformated_train.jsonl
eval_path=${DATA_ROOT}/data/yc2/annotations/original_annos/yc2_reformated_val.jsonl
eval_split_name=val

######## setup video+text features
feat_root=${DATA_ROOT}/data/yc2

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"internvideo"* ]]; then
  v_feat_dirs+=(${feat_root}/features/vid_feats)
  (( v_feat_dim += 768 ))
fi

# text features
if [[ ${t_feat_type} == "internvideo" ]]; then
  t_feat_dir=${feat_root}/features/txt_feats_original/
  t_feat_dim=1024
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

no_pin_memory=False
num_workers=8
max_q_l=-1
max_v_l=-1
max_windows=-1
num_queries=20
qd_rate=0.0

#### training
seed=$((RANDOM << 15 | RANDOM))

# Ensure PYTHONPATH includes the project root (for baselines module imports)
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}:."

PYTHONPATH=$PYTHONPATH python ld_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dir_eval ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--exp_name ${exp_name} \
--seed ${seed} \
--num_workers ${num_workers} \
--max_v_l ${max_v_l} \
--max_q_l ${max_q_l} \
--clip_length 0.3333 \
--max_windows ${max_windows} \
--num_queries ${num_queries} \
--qd_rate ${qd_rate} \
--rm_self_attn_decoder false \
${@:1}
