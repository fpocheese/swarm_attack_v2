#!/bin/bash
# V21: safety_bound=200 (above natural cost ~150)
# Key insight: With safety_bound=200, rescale_constraint_val < 0
#   → MACPO enters Case 3 (pure reward optimization)
#   → Constraint only activates when costs > 200 (genuinely dangerous)
#   → No more counterproductive constraint suppressing approach behavior
#
# Other settings: same as V20 (ap_config=none, action_scale=0.5)

cd "$(dirname "$0")/.."

echo "=== V21: MACPO safety_bound=200 (constraint-satisfied regime) ==="
echo "KEY: safety_bound=200 > natural_cost~150 → MACPO = unconstrained MAPPO"
echo "Starting training..."

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_macpo.py \
    --env_name fov_penetration \
    --algorithm_name macpo \
    --experiment_name v21_unconstrained \
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
    --safety_bound 200 \
    --ap_config none \
    2>&1 | tee outputs/v21_unconstrained_train.log

echo "Training complete."
