#!/bin/bash
# V41: Theory-grounded 22-dim observation (newswarm.tex)
#  - Self kinematic (5) | HVT LOS rate + V_c (3) | Top-2 threats q_ij/V_c/|ω|/Γ/lock (10)
#  - Team m_local + P_pen + P_hit (3) | time (1)  ⇒ obs_dim = 22
#  - No absolute positions; agent must learn PN-style guidance from LOS rate.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p outputs

echo "=========================================="
echo " V41: Theory-grounded 22-dim obs + MAPPO"
echo " Start: $(date)"
echo "=========================================="

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v41_theory_obs \
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
    --entropy_coef 0.005 \
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
    2>&1 | tee outputs/v41_theory_obs.log
