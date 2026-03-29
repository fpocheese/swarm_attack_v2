#!/bin/bash
# V11_01 Training — FOV逃逸 + 单次交汇
# 核心: 拦截器前向追踪限制, 进攻方可通过大机动打破FOV逃脱
# ⚠️ 和v11c并行训练, 不要Kill现有进程

cd /home/ps/2026/muti_uav_attack/muti_uav_attack_v11_01

export PYTHONPATH="$(pwd):$(pwd)/third_party/MACPO/MACPO:$PYTHONPATH"
mkdir -p outputs

LOGFILE="outputs/v11_01_fov_escape_$(date +%Y%m%d_%H%M%S).log"
echo "===== V11_01 FOV-Escape Training at $(date) =====" | tee "$LOGFILE"

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_macpo.py \
    --env_name fov_penetration \
    --algorithm_name macpo \
    --experiment_name v11_01_fov_escape \
    --scenario scenario_1 \
    --seed 42 \
    --cuda \
    --n_rollout_threads 20 \
    --n_training_threads 4 \
    --num_mini_batch 20 \
    --num_env_steps 10000000 \
    --episode_length 1200 \
    --ppo_epoch 5 \
    --lr 3e-4 \
    --critic_lr 3e-4 \
    --entropy_coef 0.01 \
    --hidden_size 256 \
    --layer_N 3 \
    --use_eval \
    --eval_interval 5 \
    --n_eval_rollout_threads 4 \
    --eval_episodes 8 \
    --use_linear_lr_decay \
    --log_interval 1 \
    --save_interval 10 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_param 0.2 \
    --value_loss_coef 1.0 \
    --max_grad_norm 10.0 \
    --use_max_grad_norm \
    2>&1 | tee -a "$LOGFILE"
