dset_name=hd_epic
ctx_mode=video_tef
v_feat_types=internvideo
t_feat_type=internvideo 

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data}"

results_root=${DATA_ROOT}/results/CGDETR/hd_epic_ours_debug
exp_id=exp
exp_name=txt_original

######## data paths (raw and thus overspecified annotations)
train_path=${DATA_ROOT}/data/hd_epic/annotations/original_annos/HD_EPIC_Narrations_train.jsonl
eval_path=${DATA_ROOT}/data/hd_epic/annotations/original_annos/HD_EPIC_Narrations_val.jsonl
eval_split_name=val

######## setup video+text features
feat_root=${DATA_ROOT}/data/hd_epic

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

#### training
bsz=32
enc_layers=3
dec_layers=3
t2v_layers=2
moment_layers=1
dummy_layers=2
sent_layers=1
no_pin_memory=False
num_workers=8
n_epoch=175
max_windows=-1
num_queries=50
eos_coef=0.1
qd_rate=0.25


# Ensure PYTHONPATH includes the project root (for baselines module imports)
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}:."

PYTHONPATH=$PYTHONPATH python cg_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path "${train_path}" \
--eval_path "${eval_path}" \
--eval_split_name ${eval_split_name} \
--v_feat_dirs "${v_feat_dirs[@]}" \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir "${t_feat_dir}" \
--t_feat_dir_eval "${t_feat_dir}" \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root "${results_root}" \
--exp_id ${exp_id} \
--exp_name ${exp_name} \
--enc_layers ${enc_layers} \
--dec_layers ${dec_layers} \
--t2v_layers ${t2v_layers} \
--moment_layers ${moment_layers} \
--dummy_layers ${dummy_layers} \
--sent_layers ${sent_layers} \
--num_workers ${num_workers} \
--clip_length 0.3333 \
--n_epoch ${n_epoch} \
--max_windows ${max_windows} \
--num_queries ${num_queries} \
--eos_coef ${eos_coef} \
--qd_rate ${qd_rate} \
--rm_self_attn_decoder true \
${@:1}
