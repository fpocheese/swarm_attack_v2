#!/bin/bash
# V11c Training — action_scale=0.3, 安全探索
# 核心: 策略从接近"直飞"开始微调, 而不是大幅机动

cd /home/ps/2026/muti_uav_attack/muti_uav_attack
export PYTHONPATH="$(pwd):$(pwd)/third_party/MACPO/MACPO:$PYTHONPATH"
mkdir -p outputs

LOGFILE="outputs/v11c_safescale_$(date +%Y%m%d_%H%M%S).log"
echo "===== V11c Safe-Scale Training at $(date) =====" | tee "$LOGFILE"

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_macpo.py \
    --env_name fov_penetration \
    --algorithm_name macpo \
    --experiment_name v11c_safescale \
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
    --entropy_coef 0.005 \
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
