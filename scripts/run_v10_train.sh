#!/bin/bash
# V10 Training Launch Script — optimized for RTX 3090 + 104 cores
# Usage: bash scripts/run_v10_train.sh

cd /home/ps/2026/muti_uav_attack/muti_uav_attack

export PYTHONPATH="$(pwd):$(pwd)/third_party/MACPO/MACPO:$PYTHONPATH"

mkdir -p outputs

LOGFILE="outputs/v10_train_$(date +%Y%m%d_%H%M%S).log"

echo "===== Starting V10 Training at $(date) =====" | tee "$LOGFILE"
echo "PID: $$" | tee -a "$LOGFILE"
echo "Log: $LOGFILE" | tee -a "$LOGFILE"

# V10 核心训练参数:
# - episode_length=1500: 必须≥1200 (v=50m/s, dist=6000m需要1200步)
# - n_rollout_threads=20: 20个并行环境 (104核足够)
# - hidden_size=256, layer_N=3: V10网络配置
# - num_env_steps=10000000: 10M步完整训练
# - use_eval + eval_interval=5: 每5个episode评估一次
# - use_linear_lr_decay: 学习率衰减 (runner已修复list trainer问题)

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_macpo.py \
    --env_name fov_penetration \
    --algorithm_name macpo \
    --experiment_name v10_front_sacrifice \
    --scenario scenario_1 \
    --seed 1 \
    --n_rollout_threads 20 \
    --n_training_threads 1 \
    --num_mini_batch 40 \
    --num_env_steps 10000000 \
    --episode_length 1500 \
    --ppo_epoch 5 \
    --lr 3e-4 \
    --critic_lr 3e-4 \
    --entropy_coef 0.02 \
    --hidden_size 256 \
    --layer_N 3 \
    --use_eval \
    --eval_interval 5 \
    --n_eval_rollout_threads 4 \
    --eval_episodes 8 \
    --use_linear_lr_decay \
    --log_interval 1 \
    --save_interval 20 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_param 0.2 \
    --value_loss_coef 1.0 \
    --max_grad_norm 10.0 \
    --use_max_grad_norm \
    2>&1 | tee -a "$LOGFILE"
