#!/bin/bash
# =====================================================
# V39 Training: 稳定化 - 修复V38策略发散问题
# =====================================================
# V38 → V39 关键改动 (训练诊断后):
#   V38现象: 越练越差 (reward 3700→433, eval 7373→-1070)
#            success/hit_count 始终 0, near_trigger_count 始终 0
#            entropy 1.25→1.49 上升, 策略发散
#
# V39 修复 (config.py):
#   1. lambda_heading_align       0.5  → 0.3   (cos加成减弱)
#   2. lambda_heading_error_pen   0.8  → 0.2   (过强信号是发散主因)
#   3. lambda_gamma_align         0.3  → 0.2
#   4. lambda_closing             0.8  → 0.6
#   5. lambda_mu_regularize       0.15 → 0.05  (放开探索)
#   6. hit_hvt_bonus              6000 → 3000  (减小方差)
#
# V39 训练超参:
#   - entropy_coef     0.02 → 0.005   (PPO 标准, 防止策略发散)
#   - lr               3e-4 → 1.5e-4  (更稳的更新)
#   - episode_length   8000 → 3000    (更短 episode → 更快终端反馈)
#   - clip_param       0.2  → 0.15    (更保守的策略更新)
#   - max_grad_norm    10   → 0.5     (V38=10太大, 收紧)
#   - num_mini_batch   40   → 4       (40过大, 标准 ~ 4)
#   - data_chunk_length 10  → 25      (适配 episode_length=3000)
# =====================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

mkdir -p outputs

echo "=========================================="
echo " V39: 稳定化 (修复V38策略发散)"
echo " Start: $(date)"
echo "=========================================="

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v39_stable \
    --scenario scenario_1 \
    --ap_config v28 \
    --seed 42 \
    --cuda \
    --n_rollout_threads 80 \
    --n_training_threads 8 \
    --n_eval_rollout_threads 20 \
    --num_env_steps 200000000 \
    --episode_length 3000 \
    --ppo_epoch 5 \
    --use_value_active_masks \
    --hidden_size 256 \
    --layer_N 3 \
    --lr 1.5e-4 \
    --critic_lr 1.5e-4 \
    --entropy_coef 0.005 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_param 0.15 \
    --value_loss_coef 1.0 \
    --num_mini_batch 4 \
    --max_grad_norm 0.5 \
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
    --data_chunk_length 25 \
    --use_recurrent_policy \
    2>&1 | tee outputs/v39_stable.log
