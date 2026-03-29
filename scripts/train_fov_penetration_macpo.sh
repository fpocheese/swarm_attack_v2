#!/bin/bash
# FOV Penetration MACPO 训练 V2
# 猛训练模式

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/home/uav/anaconda3/envs/rlgpu/bin/python"

export PYTHONPATH="${PROJ_ROOT}:${PROJ_ROOT}/third_party/MACPO/MACPO:${PYTHONPATH}"

cd "${PROJ_ROOT}"

${PYTHON} scripts/train_fov_penetration_macpo.py \
    --algorithm_name macpo \
    --experiment_name v2_aggressive \
    --seed 1 \
    --n_training_threads 4 \
    --n_rollout_threads 1 \
    --num_env_steps 2000000 \
    --episode_length 500 \
    --hidden_size 128 \
    --layer_N 2 \
    --lr 5e-4 \
    --critic_lr 5e-4 \
    --ppo_epoch 10 \
    --num_mini_batch 4 \
    --clip_param 0.2 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --entropy_coef 0.01 \
    --value_loss_coef 1.0 \
    --max_grad_norm 10.0 \
    --use_max_grad_norm \
    --log_interval 50 \
    --save_interval 100 \
    "$@"
