#!/bin/bash
# V11 Training — 修复坠地问题后的训练
# 改进: nz范围缩小[-0.5,1.5], gamma±15°, altitude惩罚30, smooth惩罚0.01

cd /home/ps/2026/muti_uav_attack/muti_uav_attack
export PYTHONPATH="$(pwd):$(pwd)/third_party/MACPO/MACPO:$PYTHONPATH"
mkdir -p outputs

LOGFILE="outputs/v11_anticrash_$(date +%Y%m%d_%H%M%S).log"
echo "===== V11 Anti-Crash Training at $(date) =====" | tee "$LOGFILE"
echo "Fixes: nz[-0.5,1.5], gamma±15°, alt_pen=30, smooth=0.01" | tee -a "$LOGFILE"

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_macpo.py \
    --env_name fov_penetration \
    --algorithm_name macpo \
    --experiment_name v11_anticrash \
    --scenario scenario_1 \
    --seed 1 \
    --cuda \
    --n_rollout_threads 40 \
    --n_training_threads 4 \
    --num_mini_batch 40 \
    --num_env_steps 10000000 \
    --episode_length 1500 \
    --ppo_epoch 5 \
    --lr 3e-4 \
    --critic_lr 3e-4 \
    --entropy_coef 0.01 \
    --hidden_size 256 \
    --layer_N 3 \
    --use_eval \
    --eval_interval 5 \
    --n_eval_rollout_threads 8 \
    --eval_episodes 16 \
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
