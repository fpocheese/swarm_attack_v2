#!/bin/bash
# V3 Scenario 1 (4v4) 长训练 ~7-8h
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/home/uav/anaconda3/envs/rlgpu/bin/python"
export PYTHONPATH="${PROJ_ROOT}:${PROJ_ROOT}/third_party/MACPO/MACPO:${PYTHONPATH}"
cd "${PROJ_ROOT}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PROJ_ROOT}/outputs/v3_s1_long_${TIMESTAMP}.log"

echo "Starting V3 scenario_1 long training at $(date)" | tee "$LOG_FILE"
echo "Log: $LOG_FILE"

${PYTHON} scripts/train_fov_penetration_macpo.py \
    --algorithm_name macpo \
    --experiment_name v3_scenario1_long \
    --scenario scenario_1 \
    --seed 42 \
    --n_training_threads 4 \
    --n_rollout_threads 1 \
    --num_env_steps 5000000 \
    --episode_length 500 \
    --hidden_size 256 \
    --layer_N 3 \
    --lr 3e-4 \
    --critic_lr 3e-4 \
    --ppo_epoch 15 \
    --num_mini_batch 8 \
    --clip_param 0.2 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --entropy_coef 0.01 \
    --value_loss_coef 1.0 \
    --max_grad_norm 10.0 \
    --use_max_grad_norm \
    --log_interval 50 \
    --save_interval 200 \
    2>&1 | tee -a "$LOG_FILE"

echo "Training finished at $(date)" | tee -a "$LOG_FILE"
