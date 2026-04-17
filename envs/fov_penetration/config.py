"""
FOV Penetration Environment - Configuration V21
====================================================
三维同构集群协同突防环境配置

V21 修复MACPO约束-奖励根本矛盾 (2026-03-26):

根因分析 (V19→V20 均训练失败):
  V19: safety_bound=0.1(默认), 成本~152 → rescale=(152-0.1)*0.01=+1.52
  V20: safety_bound=50,       成本~150 → rescale=(150-50)*0.01=+1.0
  两者的rescale_constraint_val都>0 → MACPO进入Case0/Case1(全力降成本!)
  但接近HVT必须穿越拦截器FOV → 成本不可降低 → 奖励梯度被约束压制 → 策略崩溃

V21 核心修复:
  safety_bound=200 (高于自然成本~150)
  → rescale=(150-200)*0.01=-0.5 < 0 → Case3(纯奖励优化!)
  → 只有成本超过200时约束才启动, 避免压制正常接近行为
  → MACPO在自然成本水平下等效于无约束MAPPO

保留V19/V20的奖励/成本配置变更(已验证合理)
  
物理分析:
  进攻方: v=60m/s, ay_max=2.5g=24.5m/s² → 最小转弯半径 R=v²/ay=146.9m
  防御方: v=65m/s, ay_max=5.0g=49.1m/s² → 最小转弯半径 R=v²/ay=86.0m
  交汇时闭合速度: ~125m/s
  在d=100m处, 进攻方2.5g法向机动:
    v_lat ≈ 12.3m/s, LOS_rate = v_lat/d = 0.123 rad/s
    要求拦截器过载: N*Vc*LOS_rate/g = 4*125*0.123/9.81 = 6.3g > 5.0g → 逃脱!
"""

import numpy as np
import copy

G = 9.81

DEFAULT_CONFIG = {
    # === 场景布局 (更紧凑, 适合单次交汇) ===
    "map_size": 2000.0,          # V23: 4000→2000m (战场缩小一半)
    "z_min": 0.0,                # V23: 100→0m (打击地面目标不能在100m判死)
    "z_max": 1000.0,             # V23: 2000→1000m
    "dt": 0.01,                  # V23: 0.05→0.01s (步长0.5m，防止飞过目标)
    "max_steps": 8000,           # V36: 6000→8000 (dt=0.01×8000=80s, 给机动留余量)

    "n_offensive": 4,
    "n_defensive": 4,

    # === 进攻飞行器参数 (V5动力学: ax/an_pitch/an_yaw控制) ===
    "offensive": {
        "v_min": 40.0,
        "v_nominal": 45.0,
        "v_max": 50.0,
        "ax_min": -5.0,          # m/s², 轴向减速下限
        "ax_max": 20.0,          # m/s², 轴向加速上限
        "an_pitch_max": 2.5 * G, # m/s², 俯仰平面法向加速度限幅 n_max=2.5
        "an_yaw_max": 2.5 * G,   # m/s², 偏航平面法向加速度限幅 n_max=2.5
        "dax_max": 60.0,         # m/s³, 轴向加速度变化率
        "dan_pitch_max": 120.0,  # m/s³, 俯仰加速度变化率
        "dan_yaw_max": 120.0,    # m/s³, 偏航加速度变化率
        "gamma_min": np.deg2rad(-15.0),
        "gamma_max": np.deg2rad(15.0),
        "action_scale": 1.0,     # V24fix: 0.5→1.0, 恢复全机动能力
    },

    # === 防御/拦截器参数 (V5动力学: ax/an_pitch/an_yaw控制) ===
    "defensive": {
        "v_min": 50.0,
        "v_nominal": 55.0,
        "v_max": 60.0,
        "ax_min": -10.0,         # m/s², 轴向减速下限
        "ax_max": 30.0,          # m/s², 轴向加速上限
        "an_pitch_max": 5.0 * G, # m/s², 俯仰平面法向加速度限幅 (约为进攻方两倍)
        "an_yaw_max": 5.0 * G,   # m/s², 偏航平面法向加速度限幅
        "dax_max": 80.0,         # m/s³, 轴向加速度变化率
        "dan_pitch_max": 150.0,  # m/s³, 俯仰加速度变化率
        "dan_yaw_max": 150.0,    # m/s³, 偏航加速度变化率
        "gamma_min": np.deg2rad(-45.0),
        "gamma_max": np.deg2rad(45.0),
    },

    # === 场景布局: V23缩小战场 (dt=0.01s，总时间60s) ===
    # 进攻方: x=-1200 → HVT: x=1200, 总攻击距离 2400m
    # 防守方: x=600, 距HVT 600m, 距进攻方 1800m
    # 50m/s飞2400m约48s, dt=0.01 max_steps=6000 共60s(有余量)
    "hvt_position": [1200.0, 0.0, 0.0],

    "offensive_init": {
        "center_x": -1200.0,
        "center_y": 0.0,
        "center_z": 300.0,
        "spread_xy": 150.0,
        "spread_z": 30.0,
        "heading_to_hvt": True,
        "heading_noise": 0.2,
        "gamma_noise": 0.02,
        "pos_noise_xy": 100.0,
        "pos_noise_z": 20.0,
    },
    "defensive_init": {
        "center_x": 600.0,
        "center_y": 0.0,
        "center_z": 350.0,
        "spread_xy": 200.0,
        "spread_z": 50.0,
        "heading_to_offense": True,
        "heading_noise": 0.3,
        "gamma_noise": 0.1,
        "pos_noise_xy": 100.0,
        "pos_noise_z": 30.0,
    },

    "fov_half_angle": np.deg2rad(30.0),
    "detection_range": 2500.0,   # V23: 覆盖整个战场(map_size=2000, 对角~2800m)

    # === V23: 对称命中机制 ===
    # kill_range: FOV锁定击杀阈值，拦截器在此范围内持续锁定N步 → 双杀
    # collision_kill_range: 纯碰撞击杀阈值（无条件双杀）
    # 两者均为3m，与 hit_hvt_range 对称（攻守均为3m命中阈值）
    "kill_range": 3.0,             # V23: 50→3m，拦截器FOV锁定击杀阈值（对称3m）
    "collision_kill_range": 5.0,   # V23: 5m CPA双杀阈值（拦截器脱靶量<5m即命中）
    "hit_hvt_range": 5.0,          # V23: 5m点目标命中（dt=0.01→步长0.5m）
    "collision_range": 5.0,

    # === V24: FOV逃逸机制已移除 — 纯碰撞击杀模式 ===
    # 拦截器逻辑: 初始分配目标 → PN制导 → 先入FOV的进攻方切换目标 → 碰撞双杀
    # 探测范围内高频位置更新，范围外低频位置更新

    # === V22: 先入视场即锁定 — 敌方规则 ===
    "enemy_lock_rules": {
        "enable_fov_trigger_lock": True,          # 先入视场即锁定
        "initial_guide_mode": "hungarian",        # 初始目指分配方式
        "lock_fov_threshold": np.deg2rad(30.0),   # 视场触发阈值 (30° half-angle)
        "lock_range_threshold": 2500.0,            # 锁定触发最大距离 (覆盖整个战场)
        "lock_persist_after_fov_loss": 200,        # V23: 20→200步 (dt=0.01s → 2s 宽限)
    },

    # === V22: 锁定后追击配置 ===
    # V31 核心改动: 拦截器飞越目标后转弯能力大幅退化
    # 这是突防场景的关键设定 — 一旦被突破就很难回头追上
    "pursuit": {
        "forward_only": False,         # 锁定后允许回头追击(但受限)
        "forward_half_angle": np.deg2rad(100.0),
        "abandon_on_pass": False,      # 锁定后不因飞过放弃
        "no_uturn": False,
        # V31: 飞越后转弯退化参数
        "uturn_ay_fraction": 0.20,     # 目标在后半球时, ay_max仅为正常的20%
        "uturn_ax_brake": -8.0,        # 回头时先减速 (m/s², 负值=减速)
        "uturn_recovery_steps": 500,   # 500步(5秒@dt=0.01)逐渐恢复全机动力
        "uturn_engage_after_pass": True,  # 飞越后仍尝试追击(但很慢)
        "passed_distance_abandon": 800.0,  # 飞越后拉开>800m → 彻底放弃
    },

    "pn_nav_gain": 4,            # V25fix: 3→4, 提高PN侧向修正能力
    "pn_direct_freq": 20.0,
    "pn_guide_freq": 5.0,        # V25fix: 2→5Hz, 保持低频但减少过大滞后
    "pn_extrapolate_horizon": 0.2,  # V26: 外推预测时间(秒), 用来补偿制导更新延迟
    # V31: 降低拦截器制导更新频率 (回头后信息延迟更大)
    "pn_guide_freq_rear": 1.0,   # 目标在后半球时, 制导更新降为1Hz

    "assignment": {
        "method": "hungarian",
        "threat_weight": 0.6,
        "intercept_cost_weight": 0.4,
        "reassign": False,            # V22: 废弃周期性重分配, 由视场触发锁定替代
        "reassign_interval": 20,      # (保留字段但无效, reassign=False)
    },

    # === V37 奖励: 修复 '飞歪' 问题 ===
    # V36诊断: Agent mu偏置0.04→航向漂移1.6°/s→50s偏80°
    #   根因: 1)heading奖只用cos(err),小角度梯度≈0 2)closing独大 3)无mu正则化
    # V37修复: 1)加heading_error平方惩罚 2)加mu正则化 3)大幅增强heading信号 4)降低closing
    "reward": {
        # --- 核心: 接近目标 ---
        "lambda_approach": 30.0,           # 距离减少奖励 (delta_rho/norm * lambda)
        "approach_norm_dist": 60.0,        # V36: 300→60 (dt=0.01步长0.45m, r_app=0.225/step)
        "close_range_threshold": 800.0,    # V37: 500→800 (更早开始放大接近奖励)
        "close_range_max_multiplier": 10.0, # 放大倍率

        # --- 核心: 航向对准 (V37大幅增强) ---
        "lambda_heading_align": 0.5,       # V37: 0.2→0.5 (cos加成更强)
        # V37新增: 航向误差平方惩罚 (小角度强梯度, 比cos有效得多)
        "lambda_heading_error_penalty": 0.8, # penalty = λ * (err/π)², 30°偏航→-0.022/step

        # --- 核心: 闭合速度 (V37削弱以停止主导) ---
        "lambda_closing": 0.8,             # V37: 1.5→0.8 (不再是最强信号)

        # --- V37新增: mu正则化 (防止无意义转弯) ---
        "lambda_mu_regularize": 0.15,      # 惩罚|action[2]|, 鼓励直飞

        # --- 核心: 距离惩罚 (防止绕圈) ---
        "lambda_proximity": 0.15,          # V37: 0.1→0.15 (略增, 配合heading强化)
        "proximity_norm_dist": 1500.0,     # 归一化距离

        # --- 安全惩罚 (仅物理) ---
        "lambda_penalty_boundary": 2.0,    # 越界
        "lambda_penalty_ground": 1.0,      # 贴地

        # --- 被杀/命中 ---
        "killed_penalty": -0.5,            # 被杀几乎不罚 → 鼓励勇敢突入
        "hit_hvt_bonus": 8000.0,           # 命中巨额奖 (V34:6000→8000)
        "step_penalty": -0.003,            # 微弱步惩罚

        # --- 终端奖励 (V35简化) ---
        "lambda_terminal_hit": 800.0,      # 每命中一架的终端奖
        "lambda_terminal_dist": 500.0,     # 距离终端奖 (越近越好)

        # --- 超时惩罚 (env.step直接引用, 必须保留) ---
        "timeout_penalty": -100.0,         # timeout基础惩罚
        "timeout_distance_penalty_coef": 1000.0, # 基于距离的timeout额外惩罚

        # V37信号量级预估 (@1000m直飞, v=45m/s, heading_err=0):
        #   approach: 30*0.45/60 = 0.225/step
        #   heading_align: 0.5*cos(0) = 0.500/step
        #   heading_err_pen: -0.8*(0/π)² = 0.000/step
        #   closing: 0.8*45/120 = 0.300/step
        #   mu_reg: -0.15*0 = 0.000/step
        #   proximity: -0.15*1000/1500 = -0.100/step
        #   NET: +0.925/step (直飞最优)
        # @1000m, 30°偏航:
        #   approach: 0.195, heading: 0.433, heading_pen: -0.022
        #   closing: 0.260, mu_reg: -0.015, proximity: -0.100
        #   NET: +0.751/step (-18.8% 相比直飞, V36仅-14%)
    },

    # V28: cost 全部并入 reward, 保留字段但不再使用
    "cost": {
        "fov_exposure": 0.0,
        "danger_zone": 0.0,
        "collision": 0.0,
        "boundary": 0.0,
        "ground_crash": 0.0,
        "speed_violation": 0.0,
    },

    # === 解析先验模块配置 (Analytic Priors) ===
    "analytic_priors": {
        # --- 全局开关 ---
        "enable_cone_cost": True,
        "enable_assignment_mismatch_reward": False,  # V22: 旧模块已废弃
        "enable_decoy_game": True,                   # V22: 新的诱饵博弈模块
        "enable_escape_reward": True,
        "enable_effective_penetration": True,          # V22: 有效突防数量

        # --- Module 1: Cone Cost ---
        "beta_cone_agg": 10.0,          # smooth-max temperature
        "M_c": 0.05,                     # terminal cone safety margin
        "cone_cost_weight": 0.03,        # V22fix: 0.15→0.03, 避免cone cost推高总cost超safety_bound
        "danger_cost_weight": 1.0,       # kappa_2
        "collision_cost_weight": 1.0,    # kappa_3
        "boundary_cost_weight": 1.0,     # kappa_4

        # Y-system parameters
        "tau_I": 0.5,                    # interceptor first-order lag time constant
        "tau_A": 0.3,                    # attacker first-order lag time constant
        "N_c": 3.0,                      # interceptor navigation constant
        "y_table_size": 500,             # Y-system lookup table resolution
        "y_integration_steps": 200,      # RK4 integration steps per t_go

        # --- Module 2: Decoy Game (replaces old assignment mismatch) ---
        "enable_decoy_game": True,       # V22: 新的诱饵博弈模块
        "k_q_sigmoid": 10.0,             # 视场占用 sigmoid 温度
        "eta_w_s": 1.0,                  # 锁定吸引: 视场占用权重
        "eta_w_rho": 0.5,                # 锁定吸引: 距离权重
        "eta_w_vc": 0.3,                 # 锁定吸引: 闭合速度权重
        "lock_prob_temperature": 5.0,    # softmax 锁定概率温度
        "decoy_self_cost_weight": 1.0,   # 诱饵自身代价权重
        "decoy_attention_benefit_weight": 1.5,  # 注意力吸引收益权重
        "decoy_team_benefit_weight": 1.0,       # 队伍突防收益权重
        "phi_decoy_weight": 0.05,        # Φ_decoy 势函数 reward 权重
        "expose_decoy_obs": True,         # 加入 observation

        # --- Module 3: LOS Escape Reward ---
        "escape_reward_weight": 0.02,    # V19: 0.05→0.02, 进一步降低避免刷触发
        "rho_trigger": 150.0,           # near-distance trigger radius (m)
        "k_rho": 0.1,                   # sigmoid smoothness parameter
        "dt_trigger": None,             # override for dt (None = use env dt)

        # --- Module 3b: Effective Penetration ---
        "enable_effective_penetration": True,
        "P_pen_cone_weight": 0.3,        # P_i_pen 中 cone safety 权重
        "P_pen_threat_weight": 0.3,      # 拦截距离权重
        "P_pen_redirect_weight": 0.2,    # 注意力重定向权重
        "P_pen_escape_weight": 0.2,      # 局部逃逸能力权重
        "kappa_h": 2.0,                  # P_i_hit miss_distance sigmoid
        "kappa_c": 1.0,                  # P_i_hit closing speed sigmoid
        "kappa_omega": 2.0,              # P_i_hit LOS rate sigmoid
        "N_eff_reward_weight": 0.3,      # N_eff 增量 reward 权重
        "N_waste_penalty_weight": 0.3,   # N_waste 惩罚权重
        "terminal_group_value_weight": 1.0,  # 终端群体价值权重
        "synergy_exponent": 1.5,         # N_eff^α 协同指数

        # --- Terminal reward params ---
        "waste_loss_weight": 0.5,        # lambda_U for N_waste penalty

        # --- Enhancement: Analytic obs in policy input ---
        "expose_analytic_obs": True,     # add Z_tilde, psi_agg, M_tilde, Xi to obs

        # --- Enhancement: HVT guidance + soft phase score ---
        "enable_hvt_guidance": True,
        "expose_hvt_guidance_obs": True,     # add rho/closing/omega/omega_dot/pn_hint/penetration_score
        "expose_penetration_share_obs": True,  # add team penetration score to share obs
        "pn_nav_gain": 3.0,
        "hvt_omega_ref": 0.6,               # rad/s normalization
        "hvt_omega_dot_ref": 0.8,           # rad/s^2 normalization
        "pn_hint_ref": 30.0,                # m/s^2 normalization
        "penetration_score_bias": -0.35,
        "penetration_score_scale": 2.2,

        # --- Enhancement: Attack-gated HVT shaping reward ---
        "enable_attack_gate_reward": True,
        "attack_gate_weight": 1.0,       # V22: 1.5→1.0, N_eff为主导, 降低attack gate
        "attack_progress_weight": 1.0,
        "attack_closing_weight": 0.8,
        "attack_los_weight": 0.5,
        "attack_losdot_weight": 0.4,
        "attack_pn_align_weight": 0.6,
        "pn_accel_ref_g": 8.0,

        # --- Enhancement: Per-agent cone cost ---
        "per_agent_cone_cost": True,     # each agent gets own psi_agg vs uniform split

        # --- Enhancement: Cooperative decoy reward ---
        "cooperative_decoy": True,       # credit individual contribution to decoy game
        "coop_decoy_individual_weight": 0.6,  # fraction attributed to causer vs team

        # --- Enhancement: Curriculum weight scheduling ---
        "curriculum_enabled": True,
        "curriculum_warmup_frac": 0.15,  # ramp from 0→full over first 15% of training
        "curriculum_total_steps": 10000000,  # total training env steps
    },

    # === 点目标命中配置 ===
    "point_target": {
        "hit_threshold": 5.0,                  # V38: 50→5m, 与拦截器碰撞阈值对称(脱靶量<5m即命中)
        "record_miss_distance": True,          # 逐步记录脱靶量
    },

    "two_stage_eval": {
        "enabled": True,
        "stage1_weight": 1.0,
        "stage2_weight": 0.3,
    },

    "obs_range": 2500.0,         # V23: 5000→2500m (匹配战场范围)
    "vel_range": 120.0,
    "z_range": 1000.0,           # V23: 2000→1000m
}

SCENARIO_CONFIGS = {
    "scenario_1": {"n_offensive": 4, "n_defensive": 4},
    "scenario_2": {"n_offensive": 4, "n_defensive": 6},
    "scenario_3": {"n_offensive": 6, "n_defensive": 4},
}


def get_config(custom_config=None, scenario=None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if scenario is not None:
        if scenario not in SCENARIO_CONFIGS:
            raise ValueError(f"Unknown scenario: {scenario}")
        for key, value in SCENARIO_CONFIGS[scenario].items():
            config[key] = value
    if custom_config is not None:
        for key, value in custom_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value
    return config
