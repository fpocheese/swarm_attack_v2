#!/bin/bash
# =====================================================
# V37 Training: Fix Heading Drift ("飞歪")
# =====================================================
# V36诊断 (飞行轨迹分析确认):
#   Agent mu偏置0.04 → heading drift 1.6°/s → 50s偏80°
#   Agent mu偏置0.20 → heading drift 7.4°/s → 60s偏444° (转好几圈!)
#   自由agent(无人追击)也飞歪 — 不是被拦截器逼的
#
# V37核心修复:
#   1. heading_error_penalty (NEW): λ=0.8, (err/π)² → 30°偏航-0.022/step
#   2. mu_regularization (NEW): λ=0.15, |action[2]| → 直接抑制转弯
#   3. heading_align: 0.2→0.5 (2.5x增强, 现在是最强正信号)
#   4. closing: 1.5→0.8 (不再主导)
#   5. close_range_threshold: 500→800m (更早放大接近奖励)
#   6. proximity: 0.1→0.15 (略增)
#
# V37信号量级 (@1000m直飞):
#   heading_align: +0.500 (最强) | approach: +0.225 | closing: +0.300
#   heading_pen: 0.000 | mu_reg: 0.000 | proximity: -0.100
#   NET: +0.925/step
# @30°偏航: NET=+0.751 (-18.8%, V36仅-14%)
#
# ⚠️  全新训练 (reward结构变化, 不resume V36)
# =====================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo " V37: Fix Heading Drift"
echo " Start: $(date)"
echo " Key: heading_err_penalty + mu_reg + heading_align 2.5x"
echo " heading_align=0.5 (最强) > closing=0.8 > approach=0.225"
echo "=========================================="

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v37_heading_fix \
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
    2>&1 | tee outputs/v37_heading_fix.log
