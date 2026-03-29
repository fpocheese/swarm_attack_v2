#!/bin/bash
# V2 Aggressive Training Launch Script
# 使用方法: nohup bash run_training.sh > /dev/null 2>&1 &

cd /home/uav/00gao_xueshu/muti_uav_attack

export PYTHONPATH="/home/uav/00gao_xueshu/muti_uav_attack:/home/uav/00gao_xueshu/muti_uav_attack/third_party/MACPO/MACPO"

LOGFILE="/home/uav/00gao_xueshu/muti_uav_attack/outputs/train_v2_aggressive.log"

echo "===== Starting V2 Aggressive Training at $(date) =====" > "$LOGFILE"
echo "PID: $$" >> "$LOGFILE"

exec /home/uav/anaconda3/envs/rlgpu/bin/python -u scripts/train_fov_penetration_macpo.py \
    --algorithm_name macpo \
    --experiment_name v2_aggressive \
    --seed 42 \
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
    --log_interval 200 \
    --save_interval 500 \
    >> "$LOGFILE" 2>&1
