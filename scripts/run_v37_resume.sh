#!/bin/bash
# =====================================================
# V37 RESUME: Continue training from checkpoint
# =====================================================
# V37 update 8 (5.76M/200M steps) was interrupted by reboot
# Diagnostics confirm heading fix is WORKING:
#   mu bias: V36=0.20 → V37=0.02 (10x reduction)
#   drift: V36=7.4°/s → V37=0.64°/s (11x reduction)
#   0 agents fly away (vs 3/4 in V36)
# Need longer training to close the distance gap.
#
# Uses screen for persistence against reboots.
# =====================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

MODEL_DIR="outputs/results/fov_penetration/mappo/v37_heading_fix/run1/models"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: No checkpoint found at $MODEL_DIR"
    echo "Run scripts/run_v37_train.sh first for fresh training."
    exit 1
fi

echo "=========================================="
echo "V37 RESUME Training"
echo "=========================================="
echo "Checkpoint: $MODEL_DIR"
echo "Time: $(date)"
echo "=========================================="

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v37_heading_fix \
    --scenario scenario_1 \
    --ap_config v28 \
    --seed 42 \
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
    --lr 3e-4 \
    --critic_lr 3e-4 \
    --entropy_coef 0.02 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_param 0.2 \
    --value_loss_coef 1.0 \
    --num_mini_batch 40 \
    --max_grad_norm 10.0 \
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
    --data_chunk_length 10 \
    --use_recurrent_policy \
    --model_dir "$MODEL_DIR" \
    2>&1 | tee -a outputs/v37_heading_fix.log
