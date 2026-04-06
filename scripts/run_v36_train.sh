#!/bin/bash
# =====================================================
# V36 Training: Fix dt=0.01 Reward Scaling
# =====================================================
# V35诊断: 训练14 updates (7.2M steps)后, 0%命中率, avg min_dist=395m
#   根因: 奖励信号失衡 (dt=0.01导致每步位移仅0.45m)
#     approach: 30*0.45/300=0.045/step (太弱!)
#     closing:  3.0*45/120=1.125/step (独大)
#     proximity: 0.4*d/1500=0.267/step@1000m (比approach大6x!)
#   + reset obs bug: first obs had zeroed HVT guidance (rho=0,closing=0)
#   + 时间不够: 6000步(60s), 直飞需53s, 仅7s余量
#
# V36修复:
#   1. approach_norm_dist: 300→60 (r_app: 0.045→0.225/step, 5x stronger)
#   2. lambda_proximity: 0.4→0.1 (@1000m: 0.267→0.067/step)
#   3. lambda_closing: 3.0→1.5 (1.125→0.563/step, 不再独大)
#   4. lambda_heading: 0.15→0.2
#   5. close_range_mult: 15→10 (避免梯度爆炸)
#   6. episode_length: 6000→8000 (60s→80s, 给机动留余量)
#   7. Fix reset obs: HVT guidance features populated on first obs
#   8. Fix dist_progress obs: correct for alive agents
#
# ⚠️  全新训练 (reward scale改变, 不能resume V35 checkpoint)
# =====================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo " V36: Fix dt=0.01 Reward Scaling"
echo " Start: $(date)"
echo " Key fixes: approach_norm 300→60, proximity 0.4→0.1"
echo "            closing 3.0→1.5, episode 6000→8000"
echo "            Fix reset obs (HVT guidance), dist_progress"
echo " Per-step signals @1000m: app=0.225 close=0.563 head=0.2 prox=-0.067"
echo "=========================================="

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v36_reward_scale_fix \
    --scenario scenario_1 \
    --ap_config v28 \
    --seed 42 \
    --cuda \
    --n_rollout_threads 80 \
    --n_training_threads 8 \
    --n_eval_rollout_threads 20 \
    --num_env_steps 200000000 \
    --episode_length 8000 \
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

