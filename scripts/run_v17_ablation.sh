#!/bin/bash
# =====================================================
# V17 Analytic Priors - Ablation Experiment Runner
# =====================================================
# This script runs 4 ablation experiments:
#   1. baseline        - original reward/cost only
#   2. +cone           - add cone cost
#   3. +cone+mismatch  - add cone cost + assignment mismatch reward
#   4. full            - all three analytic prior modules
#
# Usage:
#   bash scripts/run_v17_ablation.sh [experiment_id]
#   experiment_id: 1|2|3|4|all  (default: all)
# =====================================================

set -e
cd /home/ps/2026/muti_uav_attack/muti_uav_attack_v16

export PYTHONPATH="$(pwd):$(pwd)/third_party/MACPO/MACPO:$PYTHONPATH"
mkdir -p outputs

SEED=42
NUM_ENV_STEPS=10000000
COMMON_ARGS="
    --env_name fov_penetration
    --algorithm_name macpo
    --scenario scenario_1
    --seed ${SEED}
    --cuda
    --n_rollout_threads 20
    --n_training_threads 4
    --num_mini_batch 20
    --num_env_steps ${NUM_ENV_STEPS}
    --episode_length 1200
    --ppo_epoch 5
    --lr 3e-4
    --critic_lr 3e-4
    --entropy_coef 0.01
    --hidden_size 256
    --layer_N 3
    --use_eval
    --eval_interval 5
    --n_eval_rollout_threads 4
    --eval_episodes 8
    --use_linear_lr_decay
    --log_interval 1
    --save_interval 10
    --gamma 0.99
    --gae_lambda 0.95
    --clip_param 0.2
    --value_loss_coef 1.0
    --max_grad_norm 10.0
"

run_experiment() {
    local EXP_NAME=$1
    local AP_CONFIG=$2
    local LOGFILE="outputs/${EXP_NAME}_train.log"

    echo "===== ${EXP_NAME} Training at $(date) =====" | tee "$LOGFILE"
    echo "AP Config: ${AP_CONFIG}" | tee -a "$LOGFILE"

    nohup conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_macpo.py \
        ${COMMON_ARGS} \
        --experiment_name "${EXP_NAME}" \
        --ap_config "${AP_CONFIG}" \
        >> "$LOGFILE" 2>&1 &

    echo "PID: $!" | tee -a "$LOGFILE"
    echo "Log: $LOGFILE"
}

EXP_ID="${1:-all}"

if [[ "$EXP_ID" == "1" || "$EXP_ID" == "all" ]]; then
    echo "====== Experiment 1: Baseline (no analytic priors) ======"
    run_experiment "v17_baseline" "none"
fi

if [[ "$EXP_ID" == "2" || "$EXP_ID" == "all" ]]; then
    echo "====== Experiment 2: +Cone Cost ======"
    run_experiment "v17_cone_only" "cone"
fi

if [[ "$EXP_ID" == "3" || "$EXP_ID" == "all" ]]; then
    echo "====== Experiment 3: +Cone Cost + Mismatch Reward ======"
    run_experiment "v17_cone_mismatch" "cone_mismatch"
fi

if [[ "$EXP_ID" == "4" || "$EXP_ID" == "all" ]]; then
    echo "====== Experiment 4: Full (Cone + Mismatch + Escape) ======"
    run_experiment "v17_full" "full"
fi

echo ""
echo "All requested experiments launched. Check outputs/*_train.log for progress."
