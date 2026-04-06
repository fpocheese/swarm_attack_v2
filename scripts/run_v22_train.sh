#!/bin/bash
# V22: FOV trigger lock + decoy game + effective penetration + point target 3m
#
# V22 核心变化：
#   - 拦截机: 先入视场即锁定 (FOV trigger lock), 5 态状态机
#   - 诱饵博弈 (decoy_game): Phi_decoy potential shaping
#   - 有效突防 (effective_penetration): N_eff / N_loss / N_waste
#   - 打击阈值: 500m → 3m
#
# ap_config=v22: enable_decoy_game=True, enable_effective_penetration=True
#                enable_escape_reward=True, enable_cone_cost=True
#                enable_assignment_mismatch_reward=False (已被 decoy_game 取代)

cd "$(dirname "$0")/.."

echo "=== V22fix: safety_bound=1000 + hit_range=50m + CPA ==="
echo "ap_config=v22 | safety_bound=1000 | n_rollout_threads=20"
echo "Starting training..."

mkdir -p outputs/v22_train

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_macpo.py \
    --env_name fov_penetration \
    --algorithm_name macpo \
    --experiment_name v22fix_sb1000_hr50 \
    --scenario scenario_1 \
    --seed 42 \
    --cuda \
    --n_rollout_threads 20 \
    --n_training_threads 4 \
    --num_mini_batch 20 \
    --num_env_steps 10000000 \
    --episode_length 1200 \
    --ppo_epoch 5 \
    --lr 3e-4 \
    --critic_lr 3e-4 \
    --entropy_coef 0.02 \
    --hidden_size 256 \
    --layer_N 3 \
    --use_eval \
    --eval_interval 5 \
    --n_eval_rollout_threads 4 \
    --eval_episodes 8 \
    --use_linear_lr_decay \
    --log_interval 1 \
    --save_interval 10 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_param 0.2 \
    --value_loss_coef 1.0 \
    --max_grad_norm 10.0 \
    --safety_bound 1000 \
    --ap_config v22 \
    2>&1 | tee outputs/v22_train/v22_train.log

echo "Training complete."
