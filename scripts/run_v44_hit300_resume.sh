#!/bin/bash
# V44: 在 V43 基础上唯一改动
#   hit_hvt_range / point_target.hit_threshold: 5m → 300m (近迫毁伤半径)
#   原因 (诊断 scripts/diag_v43.py):
#     actor 输出 trim ≈ 0 → 飞机近似直线飞 → 初始 hErr~5° × 2400m ≈ 200m 几何偏差
#     5m 阈值物理上不可达 → hit_count 永远 = 0 → 没有命中梯度 → actor 永远学不到末端机动
#   300m 解锁信号后 actor 拿到正反馈即可逐步内卷收紧 min_d
# 其他参数全部沿用 V43 (低噪声 + hit-heavy reward) , 并从 v43 权重热启动
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p outputs

V43_MODEL_DIR="outputs/results/fov_penetration/mappo/v43_lownoise_hit/run1/models"

echo "=========================================="
echo " V44: hit_threshold 5m -> 300m, resume from V43 weights"
echo " Start: $(date)"
echo " Resume from: $V43_MODEL_DIR"
echo "=========================================="

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v44_hit300_resume \
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
    --lr 1.5e-4 \
    --critic_lr 1.5e-4 \
    --entropy_coef 0.002 \
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
    2>&1 | tee outputs/v44_hit300_resume.log
