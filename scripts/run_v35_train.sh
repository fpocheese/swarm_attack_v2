#!/bin/bash
# =====================================================
# V35 Training: "Target First, Fear Nothing"
# =====================================================
# V35设计理念: 之前V28-V34奖励太复杂, agent学会躲避而不是进攻.
#   用户诊断: "进攻飞行器已经有点担心自己被打击，但是不知道自己已经
#             有队友已经吸引了火力，自己只需要专心的往目标进攻即可"
#
# V35核心改动:
#   观察空间: 45→37维 (target-first, 简化threat/team/priors)
#     + heading_error_to_hvt (新)
#     + distance_progress (新)
#     + teammates_drawing_fire (新)
#     - 去掉 Gamma/Xi/Z (低价值threat信息)
#     - 简化 team summary (去掉role区分)
#     - 只保留 P_pen + P_hit (去掉 Z/Xi/Phi)
#
#   奖励系统: 20+→3个核心信号
#     1. approach_reward: lambda=30, norm=300 (极强接近奖)
#     2. heading_alignment: lambda=0.15 (面朝目标)
#     3. closing_speed: lambda=3.0 (闭合速度)
#     + proximity_penalty: lambda=0.4 (距离惩罚)
#     + hit_hvt_bonus: 8000 (巨额命中奖)
#     - 风险惩罚全部归零 (cone=0, fov=0, danger=0)
#     - 去掉decoy/escape/penetration等复杂信号
#
# ⚠️  全新训练 (obs维度变化, 无法resume旧checkpoint)
# =====================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo " V35: Target First, Fear Nothing"
echo " Start: $(date)"
echo " Fresh training (obs 45→37, new reward)"
echo " Core signals: approach(30) + heading(0.15) + closing(3.0)"
echo " Risk: ALL ZERO, killed_penalty: -0.5"
echo " Hit bonus: 8000"
echo "=========================================="

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v35_target_first \
    --scenario scenario_1 \
    --ap_config v28 \
    --seed 42 \
    --cuda \
    --n_rollout_threads 80 \
    --n_training_threads 8 \
    --n_eval_rollout_threads 20 \
    --num_env_steps 200000000 \
    --episode_length 6000 \
    --ppo_epoch 5 \
    --use_value_active_masks \
    --hidden_size 256 \
    --layer_N 3 \
    --lr 3e-4 \
    --critic_lr 3e-4 \
    --entropy_coef 0.02 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_param 0.2 \
    --value_loss_coef 1.0 \
    --num_mini_batch 40 \
    --max_grad_norm 10.0 \
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
    --data_chunk_length 10 \
    --use_recurrent_policy \
    2>&1 | tee outputs/v35_target_first.log
