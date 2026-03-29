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
  进攻方: v=50m/s, ny_max=5g → 最小转弯半径 R=v²/(g*ny)=51.0m
  防御方: v=90m/s, ny_max=8g → 最小转弯半径 R=v²/(g*ny)=103.2m
  交汇时闭合速度: ~140m/s
  在d=100m处, 进攻方5g横向机动:
    v_lat ≈ 24.5m/s, LOS_rate = v_lat/d = 0.245 rad/s
    要求拦截器过载: N*Vc*LOS_rate/g = 3*140*0.245/9.81 = 10.5g > ny_max=8g → 逃脱!
"""

import numpy as np
import copy

G = 9.81

DEFAULT_CONFIG = {
    # === 场景布局 (更紧凑, 适合单次交汇) ===
    "map_size": 4000.0,          # 缩小 (was 5000)
    "z_min": 100.0,
    "z_max": 2000.0,
    "dt": 0.1,
    "max_steps": 1200,           # 缩短 (场景更紧凑, 5000m/50m/s=100s=1000步)

    "n_offensive": 4,
    "n_defensive": 4,

    # === 进攻飞行器参数 ===
    "offensive": {
        "v_min": 40.0,
        "v_nominal": 50.0,
        "v_max": 70.0,           # 略提高 (was 65), 支持逃逸加速
        "nx_min": -0.5,
        "nx_max": 2.0,           # 提高加速能力 (was 1.5), 用于逃逸突加速
        "ny_min": -5.0,          # 提高横向机动 (was -4), 用于FOV逃逸
        "ny_max": 5.0,           # (was 4)
        "nz_min": -0.5,
        "nz_max": 1.5,
        "dnx_max": 6.0,          # 略提高 (was 5)
        "dny_max": 12.0,         # 提高变化率 (was 8), 快速拉动
        "dnz_max": 3.0,
        "gamma_min": np.deg2rad(-15.0),
        "gamma_max": np.deg2rad(15.0),
        "action_scale": 0.5,     # V20: 恢复0.5 (0.7太aggressive, 导致ground crash)
    },

    # === 防御/拦截器参数 (保持不变, 这是拦截器的硬限制) ===
    "defensive": {
        "v_min": 70.0,
        "v_nominal": 90.0,
        "v_max": 120.0,
        "nx_min": -1.0,
        "nx_max": 3.0,
        "ny_min": -8.0,          # 拦截器过载上限, 进攻方机动时需要>8g才能跟住
        "ny_max": 8.0,
        "nz_min": -3.0,
        "nz_max": 3.0,
        "dnx_max": 8.0,
        "dny_max": 15.0,
        "dnz_max": 8.0,
        "gamma_min": np.deg2rad(-45.0),
        "gamma_max": np.deg2rad(45.0),
    },

    # === 场景布局: 缩短为单次交汇设计 ===
    # 进攻方: x=-2500 → HVT: x=2500, 总攻击距离 5000m
    # 防守方: x=1200, 距HVT 1300m, 距进攻方 3700m
    # 交汇点: x = -2500 + 50*(3700/140) ≈ -1179, 距HVT约3679m
    # 交汇后: 进攻方只需飞3679m到HVT, 约73.6s
    # 拦截器无法掉头(前向追踪限制), 逃脱=突防
    "hvt_position": [2500.0, 0.0, 0.0],

    "offensive_init": {
        "center_x": -2500.0,
        "center_y": 0.0,
        "center_z": 500.0,
        "spread_xy": 300.0,
        "spread_z": 50.0,
        "heading_to_hvt": True,
        "heading_noise": 0.2,
        "gamma_noise": 0.02,
        "pos_noise_xy": 200.0,
        "pos_noise_z": 30.0,
    },
    "defensive_init": {
        "center_x": 1200.0,
        "center_y": 0.0,
        "center_z": 600.0,
        "spread_xy": 400.0,
        "spread_z": 80.0,
        "heading_to_offense": True,
        "heading_noise": 0.3,
        "gamma_noise": 0.1,
        "pos_noise_xy": 200.0,
        "pos_noise_z": 50.0,
    },

    "fov_half_angle": np.deg2rad(30.0),
    "detection_range": 2000.0,

    # === V11_01: 新型杀伤/逃逸机制 ===
    "kill_range": 50.0,            # 交战区参考半径（仅用于跟踪/脱靶逻辑，不直接击杀）
    "collision_kill_range": 3.0,   # 仅碰撞击杀阈值：脱靶量<3m 视为命中
    "hit_hvt_range": 3.0,          # V22: 点目标命中, 脱靶量 ≤ 3m 才算成功
    "collision_range": 5.0,

    # === V11_01: FOV逃逸参数 ===
    "fov_escape": {
        "enabled": True,
        "engagement_range": 200.0,     # 进入交战区的距离
        "kill_fov_required": True,     # 杀伤需要FOV锁定
        "min_tracking_steps": 3,       # 连续N步FOV锁定才能击杀
        "miss_cooldown_steps": 50,     # miss后冷却(不重新交战)
    },

    # === V22: 先入视场即锁定 — 敌方规则 ===
    "enemy_lock_rules": {
        "enable_fov_trigger_lock": True,          # 先入视场即锁定
        "initial_guide_mode": "hungarian",        # 初始目指分配方式
        "lock_fov_threshold": np.deg2rad(30.0),   # 视场触发阈值 (30° half-angle)
        "lock_range_threshold": 2000.0,            # 锁定触发最大距离
        "lock_persist_after_fov_loss": 20,         # 锁定后丢失FOV的宽限步数
    },

    # === V22: 锁定后追击配置 ===
    "pursuit": {
        "forward_only": False,         # 锁定后允许回头追击
        "forward_half_angle": np.deg2rad(100.0),
        "abandon_on_pass": False,      # 锁定后不因飞过放弃
        "no_uturn": False,
    },

    "pn_nav_gain": 3,
    "pn_direct_freq": 20.0,
    "pn_guide_freq": 2.0,

    "assignment": {
        "method": "hungarian",
        "threat_weight": 0.6,
        "intercept_cost_weight": 0.4,
        "reassign": False,            # V22: 废弃周期性重分配, 由视场触发锁定替代
        "reassign_interval": 20,      # (保留字段但无效, reassign=False)
    },

    # === 奖励 (V19: 大幅增强接近信号, 降低干扰项, 修复cost-reward失衡) ===
    "reward": {
        "hit_hvt_bonus": 6000.0,           # V19: 4000→6000, 超强终端信号
        "approach_hvt_coef": 2500.0,       # V19: 1000→2500, 前进信号必须压倒cost
        "closest_bonus_coef": 400.0,       # V19: 300→400
        "progress_coef": 1.5,              # V19: 0.5→1.5, 持续的距离进度奖励
        "proximity_reward_coef": 6.0,      # V19: 2.0→6.0, 距HVT越近每步奖励越大(指数放大)
        "retreat_penalty": -0.15,          # V19: -0.25→-0.15, 适当放松(避免过度惩罚探索)
        "mutual_kill_team_bonus": 100.0,
        "detected_penalty": -0.005,        # V19: -0.01→-0.005, 被探测是接近HVT的必然代价
        "killed_penalty": -3.0,            # V19: -5.0→-3.0, 减轻被杀惩罚(牺牲换接近是值得的)
        "step_penalty": -0.005,            # V19: -0.01→-0.005
        "timeout_penalty": -120.0,         # V19: -80→-120, 更强timeout压力
        "timeout_distance_penalty_coef": 800.0,  # V19: 600→800, timeout时按距离重罚
        "timeout_alive_penalty": -30.0,    # V19: -20→-30
        "smooth_action_coef": -0.002,      # V19: -0.005→-0.002, 减弱动作平滑惩罚
        "altitude_penalty_coef": 12.0,     # V19: 18→12, 适当放松高度惩罚
        "high_alt_penalty_coef": 5.0,      # V19: 8→5
        "spread_bonus_coef": 0.01,
        # === 距离里程碑奖励 (V19新增) ===
        "milestone_bonuses": {3000: 50.0, 2000: 100.0, 1000: 200.0, 500: 400.0},
        # === FOV逃逸奖励 ===
        "fov_escape_bonus": 200.0,         # V19: 300→200, 逃逸不应超过接近信号
        "fov_escape_team_bonus": 50.0,     # V19: 80→50
        "close_evasion_coef": 1.5,         # V19: 2.0→1.5
        "post_escape_approach_mult": 2.5,  # V19: 2.0→2.5, 逃脱后更强的前进倍率
    },

    "cost": {
        "fov_exposure": 0.3,        # V19: 1.0→0.3, 大幅降低! 被探测是接近HVT的必经之路
        "danger_zone": 1.0,        # V19: 3.0→1.0, 大幅降低! 必须穿越拦截器才能到HVT
        "collision": 5.0,
        "boundary": 2.0,
        "ground_crash": 5.0,        # V19: 10.0→5.0
        "speed_violation": 0.5,
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
        "cone_cost_weight": 0.15,        # V19: 0.6→0.15, 大幅降低cone cost避免过度保守
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
        "hit_threshold": 3.0,                  # 脱靶量 ≤ 3m 才算命中
        "record_miss_distance": True,          # 逐步记录脱靶量
    },

    "two_stage_eval": {
        "enabled": True,
        "stage1_weight": 1.0,
        "stage2_weight": 0.3,
    },

    "obs_range": 5000.0,
    "vel_range": 120.0,
    "z_range": 2000.0,
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
