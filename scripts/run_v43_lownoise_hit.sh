#!/bin/bash
# V43: 在 V42 基础上做两件事
#   1) 把动作分布的 std_y_coef 从 0.5 → 0.2 (σ: 0.366 → 0.146), 减少噪声把轨迹打死
#   2) reward shaping: 减弱 trim 白拿的 approach/closing 奖励, 放大命中信号
#      lambda_approach 30→10  lambda_closing 0.6→0.4  lambda_team_min_progress 6→4
#      hit_hvt_bonus 3000→9000  lambda_terminal_hit 800→1500  step_penalty -0.003→-0.02
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p outputs

echo "=========================================="
echo " V43: low-noise (std_y=0.2) + hit-heavy reward + 23-dim theory obs + MAPPO"
echo " Start: $(date)"
echo "=========================================="

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v43_lownoise_hit \
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
    2>&1 | tee outputs/v43_lownoise_hit.log
