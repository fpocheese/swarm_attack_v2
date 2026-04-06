#!/bin/bash
# V31 Training: 拦截器回头退化 + 强化前进奖励
# 核心改动:
#   1. 拦截器飞越目标后 ay_max 降至20%, 5秒恢复 (不再能瞬间U-turn追杀)
#   2. 奖励: lambda_hit_approach 12→18, lambda_no_retreat 4→6
#   3. 降低干扰: decoy/escape/cone/danger 惩罚全面降低
#   4. 鼓励勇敢: killed_penalty -3→-2, danger_radius 300→200

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "===== V31 Training: Interceptor UTurn Degradation + Forward-Drive Reward ====="
echo "Start time: $(date)"
echo "Changes:"
echo "  - Interceptor: uturn_ay_fraction=0.20, recovery_steps=500, ax_brake=-8"
echo "  - Reward: hit_approach=18, no_retreat=6, decoy=0.2, cone=0.3, danger=0.15"
echo "  - Timeout: -150, dist_coef=1500, terminal_hit=400"

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v28_mappo_new_obs_reward \
    --scenario scenario_1 \
    --ap_config v28 \
    --seed 42 \
    --cuda \
    --n_rollout_threads 80 \
    --n_training_threads 8 \
    --n_eval_rollout_threads 20 \
    --num_env_steps 200000000 \
    --episode_length 6000 \
    --ppo_epoch 5 \
    --use_value_active_masks \
    --hidden_size 256 \
    --layer_N 3 \
    --lr 3e-4 \
    --critic_lr 3e-4 \
    --entropy_coef 0.02 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_param 0.2 \
    --value_loss_coef 1.0 \
    --num_mini_batch 40 \
    --max_grad_norm 10.0 \
    --use_max_grad_norm \
    --use_eval \
    --eval_interval 5 \
    --eval_episodes 20 \
    --log_interval 1 \
    --save_interval 10 \
    --use_linear_lr_decay \
    --user_name fov_team \
    --use_ReLU \
    --use_feature_normalization \
    --use_orthogonal \
    --gain 0.01 \
    --data_chunk_length 10 \
    --use_recurrent_policy \
    2>&1 | tee outputs/v31_uturn_degrade.log
