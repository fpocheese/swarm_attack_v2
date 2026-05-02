#!/bin/bash
# Hourly v69 watchdog: evaluate latest checkpoint, save figures/data, and
# recover from the verified hit snapshot if the latest policy loses success.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs_remote outputs/v69_hourly_eval

LOG_FILE="logs_remote/v69_hourly_monitor.out"
FAIL_STREAK_FILE="/tmp/v69_hourly_fail_streak"
MIN_SUCCESS_EPISODES="${MIN_SUCCESS_EPISODES:-2}"
RECOVERY_FAIL_STREAK="${RECOVERY_FAIL_STREAK:-1}"
EVAL_SEEDS="${EVAL_SEEDS:-1000,1001,1002}"
VERIFIED_SNAPSHOT="outputs/results/fov_penetration/mappo/v69_hybrid_terminal_pn/run1/models_hit3_verified_snapshot"

latest_model_dir() {
    find outputs/results/fov_penetration/mappo/v69_hybrid_terminal_pn -maxdepth 2 \
        -type d -name models -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | awk 'NR==1 {print $2}'
}

ensure_tensorboard() {
    if ! ss -ltnp sport = :6006 2>/dev/null | grep -q tensorboard; then
        nohup conda run --no-capture-output -n rlgpu tensorboard \
            --logdir outputs/results/fov_penetration/mappo/v69_hybrid_terminal_pn \
            --host 127.0.0.1 --port 6006 \
            > logs_remote/tensorboard_v69.out 2>&1 < /dev/null &
        echo "[TB] relaunched pid=$!"
    fi
}

recover_from_verified() {
    if [[ ! -f "$VERIFIED_SNAPSHOT/actor_agent3.pt" ]]; then
        echo "[RECOVERY] verified snapshot missing: $VERIFIED_SNAPSHOT"
        return 0
    fi
    echo "[RECOVERY] latest checkpoint underperformed; relaunching from $VERIFIED_SNAPSHOT"
    pids=$(pgrep -f "[m]appo-fov-v69_hybrid_terminal_pn@fov_team" || true)
    if [[ -n "$pids" ]]; then
        kill -TERM $pids || true
    fi
    MODEL_DIR="$VERIFIED_SNAPSHOT" nohup bash scripts/run_v69_hybrid_terminal_pn.sh \
        > logs_remote/v69_hybrid_terminal_pn_recover_$(date +%Y%m%d_%H%M%S).out \
        2>&1 < /dev/null &
    echo "[RECOVERY] relaunched pid=$!"
}

echo "[MONITOR] started $(date) seeds=$EVAL_SEEDS min_success=$MIN_SUCCESS_EPISODES recovery_fail_streak=$RECOVERY_FAIL_STREAK" >> "$LOG_FILE"

while true; do
    {
        echo ""
        echo "[MONITOR] cycle_start $(date)"
        ensure_tensorboard
        MODEL_DIR_CURRENT="$(latest_model_dir)"
        if [[ -z "$MODEL_DIR_CURRENT" ]]; then
            MODEL_DIR_CURRENT="outputs/results/fov_penetration/mappo/v69_hybrid_terminal_pn/run1/models"
        fi
        echo "[MONITOR] model_dir=$MODEL_DIR_CURRENT"

        set +e
        EVAL_OUTPUT=$(env PYTHONUNBUFFERED=1 \
            FOV_REWARD_PROFILE=v68strictpnfix \
            FOV_OBS_PHASE_MASK=v65_strict_los \
            FOV_TERMINAL_GUIDANCE=pn_los \
            FOV_TERMINAL_PN_GAIN=3.0 \
            FOV_TERMINAL_PN_MAX_ACTION=0.8 \
            CUDA_VISIBLE_DEVICES=0 \
            conda run --no-capture-output -n rlgpu python -u scripts/eval_v69_collect.py \
                --model-dir "$MODEL_DIR_CURRENT" \
                --seeds "$EVAL_SEEDS" \
                --tag hourly 2>&1)
        EVAL_RC=$?
        set -e
        echo "$EVAL_OUTPUT"
        if [[ "$EVAL_RC" -ne 0 ]]; then
            echo "[MONITOR] eval_failed rc=$EVAL_RC"
            streak=$(cat "$FAIL_STREAK_FILE" 2>/dev/null || echo 0)
            streak=$((streak + 1))
            echo "$streak" > "$FAIL_STREAK_FILE"
            echo "[MONITOR] fail_streak=$streak"
            if [[ "$streak" -ge "$RECOVERY_FAIL_STREAK" ]]; then
                recover_from_verified
                echo 0 > "$FAIL_STREAK_FILE"
            fi
        else
            SUMMARY_JSON="$(echo "$EVAL_OUTPUT" | tail -n 1)"
            PARSED_COUNTS=$(SUMMARY_JSON="$SUMMARY_JSON" conda run --no-capture-output -n rlgpu python - <<'PY'
import json
import os

try:
    data = json.loads(os.environ["SUMMARY_JSON"])
    print(int(data.get("success_episodes", 0)), int(data.get("total_hits", 0)))
except Exception:
    print(0, 0)
PY
)
            PARSED_COUNTS=$(echo "$PARSED_COUNTS" | tail -n 1)
            SUCCESS_EPISODES=$(echo "$PARSED_COUNTS" | awk '{print $1}')
            TOTAL_HITS=$(echo "$PARSED_COUNTS" | awk '{print $2}')
            SUCCESS_EPISODES="${SUCCESS_EPISODES:-0}"
            TOTAL_HITS="${TOTAL_HITS:-0}"
            echo "[MONITOR] success_episodes=$SUCCESS_EPISODES total_hits=$TOTAL_HITS"
            if [[ "$SUCCESS_EPISODES" -lt "$MIN_SUCCESS_EPISODES" ]]; then
                streak=$(cat "$FAIL_STREAK_FILE" 2>/dev/null || echo 0)
                streak=$((streak + 1))
                echo "$streak" > "$FAIL_STREAK_FILE"
                echo "[MONITOR] fail_streak=$streak"
                if [[ "$streak" -ge "$RECOVERY_FAIL_STREAK" ]]; then
                    recover_from_verified
                    echo 0 > "$FAIL_STREAK_FILE"
                fi
            else
                echo 0 > "$FAIL_STREAK_FILE"
            fi
        fi
        echo "[MONITOR] cycle_end $(date)"
    } >> "$LOG_FILE" 2>&1
    sleep 3600
done