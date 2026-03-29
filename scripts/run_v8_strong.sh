#!/bin/bash
export PYTHONPATH="/home/uav/00gao_xueshu/muti_uav_attack:/home/uav/00gao_xueshu/muti_uav_attack/third_party/MACPO/MACPO"
cd /home/uav/00gao_xueshu/muti_uav_attack
mkdir -p outputs/logs

echo "[$(date)] Starting V8 strong-approach training (5M steps, ~2.5hrs)"

/home/uav/anaconda3/envs/rlgpu/bin/python scripts/train_fov_penetration_macpo.py \
    --algorithm_name macpo \
    --experiment_name v8_strong_approach \
    --scenario scenario_1 \
    --seed 42 \
    --n_training_threads 4 \
    --n_rollout_threads 20 \
    --num_env_steps 5000000 \
    --episode_length 500 \
    --hidden_size 256 \
    --layer_N 3 \
    --lr 3e-4 \
    --critic_lr 3e-4 \
    --ppo_epoch 10 \
    --num_mini_batch 8 \
    --clip_param 0.2 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --entropy_coef 0.05 \
    --value_loss_coef 1.0 \
    --max_grad_norm 10.0 \
    --use_max_grad_norm \
    --log_interval 20 \
    --save_interval 100

echo "[$(date)] V8 Training finished (exit=$?)"
