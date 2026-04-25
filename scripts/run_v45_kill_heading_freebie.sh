#!/bin/bash
# V45: kill trim white-leech via lambda_heading_align 0.3 -> 0.05
#   Hypothesis: V44 actor trims (avg_act ~ 0) because spawn heading already
#   points at HVT, so heading_align=0.3 pays +0.3/step * 8000 ~= +2400 for free,
#   dominating end-game dense bonus. Cut 6x; heading still defended by
#   lambda_heading_error_penalty=0.2.
#   Env frozen (dt=0.01, hit=5m, max_steps=8000), only config.py reward block changed.
#   Resume from v44_remote_fresh latest run (run3) to preserve early skills.
source ~/miniconda3/etc/profile.d/conda.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p outputs logs_remote

V44_MODEL_DIR="outputs/results/fov_penetration/mappo/v44_remote_fresh/run3/models"
if [[ ! -d "$V44_MODEL_DIR" ]] || [[ ! -f "$V44_MODEL_DIR/actor_agent3.pt" ]]; then
    echo "[FATAL] V44 model dir incomplete: $V44_MODEL_DIR"
    exit 1
fi

echo "=========================================="
echo " V45: kill heading_align freebie (0.3 -> 0.05)"
echo " Resume from: $V44_MODEL_DIR"
echo " Start: $(date)"
echo "=========================================="

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v45_kill_heading_freebie \
    --scenario scenario_1 \
    --ap_config v28 \
    --cuda \
    --n_rollout_threads 80 \
    --n_training_threads 8 \
    --n_eval_rollout_threads 20 \
    --num_env_steps 200000000 \
    --episode_length 8000 \
    --ppo_epoch 5 \
    --use_value_active_masks \
    --hidden_size 256 \
    --layer_N 3 \
    --lr 1.0e-4 \
    --critic_lr 1.0e-4 \
    --entropy_coef 0.003 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_param 0.15 \
    --value_loss_coef 1.0 \
    --num_mini_batch 8 \
    --max_grad_norm 0.5 \
    --use_max_grad_norm \
    --use_eval \
    --eval_interval 5 \
    --eval_episodes 20 \
    --log_interval 1 \
    --save_interval 10 \
    --use_linear_lr_decay \
    --user_name fov_team \
    --use_ReLU \
    --use_feature_normalization \
    --use_orthogonal \
    --gain 0.01 \
    --data_chunk_length 25 \
    --use_recurrent_policy \
    --std_x_coef 1.0 \
    --std_y_coef 0.2 \
    --seed 4500 \
    --model_dir "$V44_MODEL_DIR" \
    2>&1 | tee outputs/v45_kill_heading_freebie.log
