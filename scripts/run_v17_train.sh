#!/bin/bash
# =====================================================
# V17 Full Training — All Analytic Priors Enabled
# =====================================================
# Cone Cost + Assignment Mismatch Reward + LOS Escape Reward
# =====================================================

cd /home/ps/2026/muti_uav_attack/muti_uav_attack_v16

export PYTHONPATH="$(pwd):$(pwd)/third_party/MACPO/MACPO:$PYTHONPATH"
mkdir -p outputs

LOGFILE="outputs/v17_full_train.log"
echo "===== V17 Full Analytic Priors Training at $(date) =====" | tee "$LOGFILE"

nohup conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_macpo.py \
    --env_name fov_penetration \
    --algorithm_name macpo \
    --experiment_name v17_full \
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
    --entropy_coef 0.01 \
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
    --ap_config full \
    >> "$LOGFILE" 2>&1 &

echo "PID: $!" | tee -a "$LOGFILE"
echo "Training started. Monitor with: tail -f $LOGFILE"
