#!/bin/bash
# V15 Training — 基于V11_01最佳版本 + 增强前进信号/高度保护
# 
# V15 核心改进 (vs V11_01):
#   1. hit_hvt_bonus: 2000→4000 (更强终端信号)
#   2. 新增proximity_reward: 距HVT越近每步奖励越大
#   3. approach_hvt_coef: 800→1000 (更强前进动力)  
#   4. retreat_penalty: -0.15→-0.25 (更严惩后退)
#   5. gamma惩罚收紧: 俯冲-5°→-3°, 爬升10°→8° 触发
#   6. 安全高度收紧: 300-800m, +0.03/step
#   7. fov_escape_bonus: 200→300 (更强逃逸信号)
#   8. timeout_penalty: -50→-80

cd /home/ps/2026/muti_uav_attack/muti_uav_attack_v15

export PYTHONPATH="$(pwd):$(pwd)/third_party/MACPO/MACPO:$PYTHONPATH"
mkdir -p outputs

LOGFILE="outputs/v15_train.log"
echo "===== V15 Training at $(date) =====" | tee "$LOGFILE"
echo "Key changes: hit_hvt=4000, approach=1000, proximity=2.0, retreat=-0.25, gamma tighter, safe_alt 300-800m" | tee -a "$LOGFILE"

nohup conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_macpo.py \
    --env_name fov_penetration \
    --algorithm_name macpo \
    --experiment_name v15_enhanced \
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
    >> "$LOGFILE" 2>&1 &

echo "V15 training PID: $!" | tee -a "$LOGFILE"
echo "Monitor: tail -f $LOGFILE"
