#!/bin/bash
# =====================================================
# V33 Training: Resume from run10 checkpoint with reward fixes
# =====================================================
# 核心改动:
#   1. 从 run10 checkpoint 恢复训练 (update 38/416处中断)
#   2. 奖励修复: 近距指数放大(500m内最高20倍), 强化终端命中
#   3. hit_threshold 30m→50m (更容易触发命中奖励)
#   4. 降低避险惩罚(cone/fov/danger), 鼓励勇敢突入
#   5. 加强timeout惩罚(-200) + 远离距离惩罚(2000)
# =====================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

MODEL_DIR="outputs/results/fov_penetration/mappo/v28_mappo_new_obs_reward/run10/models"

echo "===== V33 Training: Resume from run10 + Near-Distance Exponential Reward Fix ====="
echo "Start time: $(date)"
echo "Resuming from: $MODEL_DIR"
echo "Changes:"
echo "  - Reward: hit_approach=22, close_range_threshold=500m, max_multiplier=20x"
echo "  - Less fear: cone=0.2, fov=0.03, danger=0.1, danger_radius=150"
echo "  - Stronger drive: no_retreat=8, timeout=-200, dist_coef=2000"
echo "  - hit_threshold: 50m, hit_hvt_bonus: 6000"
echo "  - terminal: eff=60, hit=500, synergy=25, loss=6, waste=8"

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
    2>&1 | tee outputs/v33_resume_close_range_fix.log
