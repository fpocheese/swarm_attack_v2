#!/bin/bash
export PYTHONPATH="/home/uav/00gao_xueshu/muti_uav_attack:/home/uav/00gao_xueshu/muti_uav_attack/third_party/MACPO/MACPO"
cd /home/uav/00gao_xueshu/muti_uav_attack

echo "[$(date)] Starting V3 scenario_1 nz-fix training (800K steps, 20 rollout threads)"

/home/uav/anaconda3/envs/rlgpu/bin/python scripts/train_fov_penetration_macpo.py \
    --algorithm_name macpo \
    --experiment_name v3_scenario1_nzfix \
    --scenario scenario_1 \
    --seed 88 \
    --n_training_threads 4 \
    --n_rollout_threads 20 \
    --num_env_steps 800000 \
    --episode_length 500 \
    --hidden_size 256 \
    --layer_N 3 \
    --lr 5e-4 \
    --critic_lr 5e-4 \
    --ppo_epoch 15 \
    --num_mini_batch 8 \
    --clip_param 0.2 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --entropy_coef 0.01 \
    --value_loss_coef 1.0 \
    --max_grad_norm 10.0 \
    --use_max_grad_norm \
    --log_interval 20 \
    --save_interval 100

TRAIN_EXIT=$?
echo "[$(date)] Training finished (exit=$TRAIN_EXIT)"

# 训练完毕后自动评估 + 诊断
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "[$(date)] Starting auto-evaluation..."
    
    # 找到最新的run目录
    LATEST_RUN=$(ls -1d /home/uav/00gao_xueshu/muti_uav_attack/outputs/results/fov_penetration/macpo/v3_scenario1_nzfix/run* 2>/dev/null | sort -V | tail -1)
    echo "Latest run: $LATEST_RUN"
    
    if [ -d "$LATEST_RUN/models" ]; then
        # 创建评估脚本的临时配置
        /home/uav/anaconda3/envs/rlgpu/bin/python -c "
import sys, os
sys.path.insert(0, '/home/uav/00gao_xueshu/muti_uav_attack')
sys.path.insert(0, '/home/uav/00gao_xueshu/muti_uav_attack/third_party/MACPO/MACPO')

# 修改eval脚本指向新模型
model_dir = '$LATEST_RUN/models'
print(f'Evaluating model: {model_dir}')

# 复用diagnose脚本
exec(open('scripts/diagnose_episode.py').read().replace(
    \"'v3_scenario1_fast', 'run3'\",
    \"'v3_scenario1_nzfix', '\" + os.path.basename('$LATEST_RUN') + \"'\"
).replace(
    \"'fast_run3_ep00'\",
    \"'nzfix_ep00'\"
))
"
        echo "[$(date)] Diagnostics complete!"
    fi
fi
