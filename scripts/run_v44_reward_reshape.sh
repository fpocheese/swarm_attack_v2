#!/bin/bash
# V44: 反 trim 白嫖 reward 改造 (env 完全冻结: dt=0.01, hit=5m, max_steps=8000)
#   配置改动 (envs/fov_penetration/config.py reward 块):
#     - lambda_approach 10 → 3
#     - lambda_closing 0.4 → 0.15
#     - lambda_proximity 0.15 → 0.05
#     - 新增 lambda_proximity_dense=8.0, sigma=200m  (exp(-d/σ) 末端稠密引导)
#     - 新增 lambda_overshoot=5.0, trigger=800m       (过冲重罚)
#     - lambda_terminal_dist 500 → 1000, 改为 exp(-min_d/80) 反逼末端精度
#   预期: actor 不再因 trim 直飞拿大头, 必须末端机动收 d 才有正回报
#   resume 自 V43 权重以保留早期航向对准技能
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p outputs

V43_MODEL_DIR="outputs/results/fov_penetration/mappo/v43_lownoise_hit/run1/models"
if [[ ! -d "$V43_MODEL_DIR" ]]; then
    echo "[FATAL] V43 model dir not found: $V43_MODEL_DIR"
    exit 1
fi

echo "=========================================="
echo " V44: reward reshape (anti-trim + overshoot + exp dense + exp terminal)"
echo " Resume from: $V43_MODEL_DIR"
echo " Start: $(date)"
echo "=========================================="

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v44_reward_reshape \
    --scenario scenario_1 \
    --ap_config v28 \
    --seed 44 \
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
    --lr 1.0e-4 \
    --critic_lr 1.0e-4 \
    --entropy_coef 0.003 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_param 0.15 \
    --value_loss_coef 1.0 \
    --num_mini_batch 8 \
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
    --std_x_coef 1.0 \
    --std_y_coef 0.2 \
    --model_dir "$V43_MODEL_DIR" \
    2>&1 | tee outputs/v44_reward_reshape.log
