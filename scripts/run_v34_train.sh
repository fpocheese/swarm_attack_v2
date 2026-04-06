#!/bin/bash
# =====================================================
# V34 Training: Fix wrong-file bug + Reward scale rebuild
# =====================================================
# V33失败根因: reward_cost_v28.py was modified but env imports reward_cost.py (V29)
# V34修复:
#   1. 直接修改reward_cost.py:
#      - approach归一化: obs_range(2500) → approach_norm_dist(500)   [5x stronger]
#      - 新增per-step proximity drive: lambda=0.3                    [distance penalty]
#      - V33指数近距放大实际生效                                       [was never applied!]
#   2. Config:
#      - Risk大幅降低: cone=0.1, fov=0.01, danger=0.05
#      - danger_radius: 120m
#      - killed_penalty: -1.0
#   3. 验证:
#      - V34 approach/step at 1000m: 0.123 (was 0.025)  → 5x stronger
#      - V34 risk/step: 0.118 (was 0.263)  → 2.2x weaker
#      - 接近信号首次 > 风险信号
# =====================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

MODEL_DIR="outputs/results/fov_penetration/mappo/v28_mappo_new_obs_reward/run11/models"

echo "=========================================="
echo " V34 Training (fix wrong-file + reward scale)"
echo " Start time: $(date)"
echo " Resume from: $MODEL_DIR"
echo " Key changes:"
echo "   - approach_norm_dist: 2500 -> 500 (5x stronger approach)"
echo "   - proximity drive: lambda=0.3 (new per-step penalty)"
echo "   - risk: cone 0.1, fov 0.01, danger 0.05"
echo "   - close-range exponential NOW APPLIED (was in wrong file)"
echo "=========================================="

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v28_mappo_new_obs_reward \
    --scenario scenario_1 \
    --ap_config v28 \
    --seed 42 \
    --cuda \
    --model_dir "$MODEL_DIR" \
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
    2>&1 | tee outputs/v34_reward_scale_fix.log
