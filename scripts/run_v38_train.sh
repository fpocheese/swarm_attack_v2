#!/bin/bash
# =====================================================
# V38 Training: V5惯性系加速度 + 排他锁定 + 5m命中阈值
# =====================================================
# V38 核心改动 (2026-04-08):
#   1. 动力学: (ax, ay, mu) → (ax, an_pitch, an_yaw) 惯性系加速度
#      - action=[0,0,0] → 平飞trim (ax=0, an_pitch=g, an_yaw=0)
#      - 进攻方: 2.5g加速度上限, 防御方: 5.0g
#   2. 排他锁定: 每个进攻飞行器最多被1个拦截器锁定
#   3. 命中阈值: HVT命中 50m→5m (与拦截器碰撞阈值对称)
#   4. 拦截器PN制导: 直接输出an_pitch/an_yaw, 不再经过合成
#
# 训练目标:
#   进攻飞行器(RL) vs 拦截器(PN) — 突防打击HVT
#
# PN-vs-PN验证结果:
#   ✓ 平飞稳定 (action=[0,0,0], 零高度漂移)
#   ✓ 拦截器PN命中进攻方 (脱靶量4.5m < 5m)
#   ✓ 排他锁定无违规
# =====================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo " V38: 惯性系加速度 + 排他锁定 + 5m命中"
echo " Start: $(date)"
echo " 动力学: (ax, an_pitch, an_yaw) 惯性系"
echo " 进攻方RL vs 拦截器PN"
echo "=========================================="

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v38_inertial_accel \
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
    2>&1 | tee outputs/v38_inertial_accel.log
