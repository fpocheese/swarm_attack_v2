#!/bin/bash
export PYTHONPATH="/home/uav/00gao_xueshu/muti_uav_attack:/home/uav/00gao_xueshu/muti_uav_attack/third_party/MACPO/MACPO"
cd /home/uav/00gao_xueshu/muti_uav_attack

mkdir -p outputs/logs

echo "[$(date)] Starting V3 scenario_1 FAST training (~800K steps)"

/home/uav/anaconda3/envs/rlgpu/bin/python scripts/train_fov_penetration_macpo.py \
    --algorithm_name macpo \
    --experiment_name v3_scenario1_fast \
    --scenario scenario_1 \
    --seed 66 \
    --n_training_threads 4 \
    --n_rollout_threads 20 \
    --num_env_steps 800000 \
    --episode_length 500 \
    --hidden_size 256 \
    --layer_N 3 \
    --lr 5e-4 \
    --critic_lr 5e-4 \
    --ppo_epoch 15 \
    --num_mini_batch 8 \
    --clip_param 0.2 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --entropy_coef 0.01 \
    --value_loss_coef 1.0 \
    --max_grad_norm 10.0 \
    --use_max_grad_norm \
    --log_interval 20 \
    --save_interval 100

echo "[$(date)] Training finished"
