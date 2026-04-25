"""
FOV Penetration Environment V11_01
=====================================
三维同构集群协同突防环境

V11_01 核心改进:
  1. FOV跟踪式杀伤: 拦截器需要在杀伤区内保持FOV锁定才能击杀
  2. FOV逃逸机制: 进攻方突破FOV锁定 → 触发miss → 拦截器放弃
  3. 单次交汇: 拦截器前向追踪限制, 目标飞过后不掉头
  4. 过载饱和: PN需求过载超出拦截器极限时自然产生miss
"""

import numpy as np
from gym.spaces import Box

from .config import get_config, G
from .entities import Aircraft, HVT
from .dynamics import action_to_control_3d
from .reward_cost import compute_rewards, compute_costs, compute_terminal_rewards
from .policies_interceptor import InterceptorPolicy
from .target_assignment import assign_targets
from .analytic_priors import (
    YSystemCache,
    compute_group_cone_cost,
    compute_initial_assignment,
    compute_assignment_mismatch,
    compute_escape_reward,
    compute_hvt_guidance_features,
    compute_penetration_success_score,
    compute_decoy_game,
    compute_effective_penetration,
)


class FOVPenetrationEnv:

    def __init__(self, config=None, scenario=None):
        self.config = get_config(config, scenario=scenario)
        cfg = self.config
        self.n_offensive = cfg["n_offensive"]
        self.n_defensive = cfg["n_defensive"]
        self.n_agents = self.n_offensive

        self.hvt = HVT(cfg["hvt_position"][0], cfg["hvt_position"][1],
                       cfg["hvt_position"][2] if len(cfg["hvt_position"]) > 2 else 0.0)

        self.offensives = []
        self.defensives = []
        self.defensive_policies = []
        self._compute_space_dims()

        self.observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(self.share_obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.action_space = tuple([
            Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
            for _ in range(self.n_agents)
        ])

        self.current_step = 0
        self.max_steps = cfg["max_steps"]
        self.dt = cfg["dt"]
        self._seed = None
        self.rng = np.random.RandomState()
        self.prev_dists_to_hvt = []
        self.initial_dists_to_hvt = []
        self.hit_count = 0
        self.hit_indices = []
        self.kill_events = []
        self.escape_events_total = []
        self.assignments = {}
        self.lock_on_map = {}
        self.prev_team_min_dist = None

        # V22: 显式锁定映射 (youneedread 2.2)
        self.locked_target_by_defender = {}  # {def_idx: off_idx or None}
        self.locked_by_map = {}              # {off_idx: [def_idx, ...]}
        self.lock_events_log = []            # 全局锁定事件日志

        # 交战跟踪
        self.engagement_tracking = {}
        self.miss_cooldowns = {}

        # ====== Analytic Priors State ======
        self.ap_config = cfg.get("analytic_priors", {})
        self._ap_enabled = (self.ap_config.get("enable_cone_cost", False)
                            or self.ap_config.get("enable_assignment_mismatch_reward", False)
                            or self.ap_config.get("enable_escape_reward", False)
                            or self.ap_config.get("enable_hvt_guidance", False)
                            or self.ap_config.get("enable_decoy_game", False)
                            or self.ap_config.get("enable_effective_penetration", False))
        # Y-system cache (built once, shared across episodes)
        if self.ap_config.get("enable_cone_cost", False) or self.ap_config.get("enable_assignment_mismatch_reward", False):
            self._y_cache = YSystemCache(
                tau_I=self.ap_config.get("tau_I", 0.5),
                tau_A=self.ap_config.get("tau_A", 0.3),
                N_c=self.ap_config.get("N_c", 3.0),
                t_go_max=(cfg["max_steps"] + 10) * cfg["dt"],
                n_table=self.ap_config.get("y_table_size", 500),
                n_integration_steps=self.ap_config.get("y_integration_steps", 200),
            )
        else:
            self._y_cache = None

        # V28: Analytic priors per-episode state (initialized in reset)
        self._ap_prev_q_matrix = None
        self._ap_fixed_assignment = {}
        self._ap_prev_M_tilde = None
        self._ap_prev_hvt_omega = np.zeros(self.n_offensive, dtype=np.float32)
        self._ap_global_step = 0
        self._ap_prev_Phi_decoy = None
        self._ap_prev_N_eff = None

        # V28: New analytic data caches for obs building
        self._ap_decoy_info = {}
        self._ap_pen_info = {}
        self._ap_esc_info = {}
        self._ap_hvt_info = {}
        self._ap_Z_matrix = np.zeros((self.n_offensive, self.n_defensive), dtype=np.float32)
        self._ap_Z_tilde = np.zeros(self.n_offensive, dtype=np.float32)
        self._ap_psi_agg = np.zeros(self.n_offensive, dtype=np.float32)
        self._ap_Gamma_matrix = [[0.0] * self.n_defensive for _ in range(self.n_offensive)]
        self._ap_Xi_matrix = [[0.0] * self.n_defensive for _ in range(self.n_offensive)]
        self._ap_prev_E_esc = None

    def _compute_space_dims(self):
        """V41 观测空间: 理论驱动 (Theory-Grounded), 全部相对量/理论量, 无绝对位置.

        理论依据 (newswarm.tex):
          - 拦截器: q_ij (FOV锥裕度), V_c_ij (闭合速度), |omega_LOS_ij| (视线角速度),
                   Gamma_ij = |omega_LOS| - a_max/V_j (跟踪失配裕度), is_locking_me
          - HVT:    LOS azimuth/elevation rate (dλ_az/dt, dλ_el/dt) + V_c + heading-LOS alignment cos
                   (PN 制导核心: dλ/dt 指明应施加的横/纵向加速度方向; cos 给纯方向引导, 不含距离)
          - 群体:   被锁定队友数 (注意力转移 m_local), P_pen_i, P_hit_i (理论先验)

        分块:
          1. Self kinematic (5):   v, sin/cos(gamma), an_pitch, an_yaw   — 无绝对位置
          2. HVT guidance (4):     dλ_az/dt, dλ_el/dt, V_c_HVT, cos(heading, r_iH)
          3. Top-K threats (5*K):  q_ij, V_c_ij, |omega_LOS_ij|, Gamma_ij, is_locking_me
          4. Team & priors (3):    n_mates_drawing_fire, P_pen_i, P_hit_i
          5. Time (1):             t/T
        Total (K=2): 5 + 4 + 10 + 3 + 1 = 23
        """
        cfg = self.config
        n_off = cfg["n_offensive"]
        self.obs_k_def = min(cfg["n_defensive"], 4)

        self._dim_self = 5
        self._dim_hvt_guidance = 4
        self._dim_per_threat = 5      # q_ij, V_c, |omega|, Gamma, is_locking
        self._n_top_threats = 2
        self._dim_threats = self._dim_per_threat * self._n_top_threats  # 10
        self._dim_team_priors = 3     # n_mates_drawing_fire, P_pen, P_hit
        self._dim_time = 1

        self.obs_dim = (self._dim_self + self._dim_hvt_guidance
                        + self._dim_threats + self._dim_team_priors
                        + self._dim_time)  # 23

        # share obs: 所有进攻方 10维 + 所有防御方 7维 + HVT 3维 + global 5维 + team_pen 1维
        self.share_obs_dim = 10 * n_off + 7 * cfg["n_defensive"] + 3 + 5 + 1

        self._ap_obs_dim = 0

    def seed(self, seed=None):
        self._seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed)

    def _create_entities(self):
        cfg = self.config
        off_params = cfg["offensive"]
        def_params = cfg["defensive"]
        oi = cfg["offensive_init"]
        di = cfg["defensive_init"]

        hvt_x, hvt_y = cfg["hvt_position"][0], cfg["hvt_position"][1]
        base_heading = np.arctan2(hvt_y - oi["center_y"], hvt_x - oi["center_x"])

        self.offensives = []
        for i in range(self.n_offensive):
            angle = 2 * np.pi * i / self.n_offensive
            x = oi["center_x"] + oi["spread_xy"] * np.cos(angle) + self.rng.uniform(-oi["pos_noise_xy"], oi["pos_noise_xy"])
            y = oi["center_y"] + oi["spread_xy"] * np.sin(angle) + self.rng.uniform(-oi["pos_noise_xy"], oi["pos_noise_xy"])
            z = oi["center_z"] + self.rng.uniform(-oi["pos_noise_z"], oi["pos_noise_z"])
            z = np.clip(z, cfg["z_min"], cfg["z_max"])
            heading = base_heading + self.rng.uniform(-oi["heading_noise"], oi["heading_noise"])
            gamma = self.rng.uniform(-oi["gamma_noise"], oi["gamma_noise"])
            off = Aircraft(i, "offensive", off_params,
                           x=x, y=y, z=z, v=off_params["v_nominal"],
                           heading=heading, gamma=gamma)
            # V22: 逃逸/锁定标记已在 entities.py V4 reset() 中初始化
            self.offensives.append(off)

        atk_cx = np.mean([o.x for o in self.offensives])
        atk_cy = np.mean([o.y for o in self.offensives])
        def_base_heading = np.arctan2(atk_cy - di["center_y"], atk_cx - di["center_x"])

        self.defensives = []
        self.defensive_policies = []
        for i in range(self.n_defensive):
            angle = 2 * np.pi * i / self.n_defensive
            x = di["center_x"] + di["spread_xy"] * np.cos(angle) + self.rng.uniform(-di["pos_noise_xy"], di["pos_noise_xy"])
            y = di["center_y"] + di["spread_xy"] * np.sin(angle) + self.rng.uniform(-di["pos_noise_xy"], di["pos_noise_xy"])
            z = di["center_z"] + self.rng.uniform(-di["pos_noise_z"], di["pos_noise_z"])
            z = np.clip(z, cfg["z_min"], cfg["z_max"])
            heading = def_base_heading + self.rng.uniform(-di["heading_noise"], di["heading_noise"])
            gamma = self.rng.uniform(-di["gamma_noise"], di["gamma_noise"])
            d = Aircraft(100 + i, "defensive", def_params,
                         x=x, y=y, z=z, v=def_params["v_nominal"],
                         heading=heading, gamma=gamma)
            self.defensives.append(d)
            policy = InterceptorPolicy(d, self.hvt, cfg, patrol_idx=i)
            self.defensive_policies.append(policy)

    def _run_target_assignment(self):
        """V22: 仅在 reset() 中调用一次, 提供初始目指"""
        self.assignments, _, _ = assign_targets(
            self.defensives, self.offensives, self.hvt, self.config)
        for def_idx, off_idx in self.assignments.items():
            if def_idx < len(self.defensive_policies):
                self.defensive_policies[def_idx].set_initial_target(
                    off_idx, self.offensives[off_idx])

    def _update_lock_on_map(self):
        """V22: 更新锁定映射 (基于实际 FOV 触发锁定状态)"""
        self.lock_on_map = {i: [] for i in range(self.n_offensive)}
        self.locked_target_by_defender = {}
        self.locked_by_map = {i: [] for i in range(self.n_offensive)}

        for def_idx, policy in enumerate(self.defensive_policies):
            d = self.defensives[def_idx]
            if not d.alive:
                self.locked_target_by_defender[def_idx] = None
                continue

            # V22: 使用锁定状态机的 current_locked_target_idx
            if policy.lock_mode == InterceptorPolicy.STATE_LOCKED:
                off_idx = policy.current_locked_target_idx
                if off_idx is not None and off_idx < self.n_offensive:
                    off = self.offensives[off_idx]
                    if off.alive and not off.hit_hvt:
                        self.locked_target_by_defender[def_idx] = off_idx
                        self.locked_by_map[off_idx].append(def_idx)
                        self.lock_on_map[off_idx].append(def_idx)
                        # 更新 entities 中的锁定信息
                        off.locked_by_defenders = self.locked_by_map[off_idx]
                        off.locked_by_count = len(self.locked_by_map[off_idx])
                        continue
            # INIT_GUIDE 阶段: 也维护 lock_on_map (用初始分配目标)
            elif policy.lock_mode == InterceptorPolicy.STATE_INIT_GUIDE:
                off_idx = policy.initial_assigned_target_idx
                if off_idx is not None and off_idx < self.n_offensive:
                    self.lock_on_map[off_idx].append(def_idx)

            self.locked_target_by_defender[def_idx] = None

        # 更新所有进攻方的 locked_by 信息
        for i, off in enumerate(self.offensives):
            off.locked_by_defenders = self.locked_by_map.get(i, [])
            off.locked_by_count = len(off.locked_by_defenders)

    def reset(self):
        self.current_step = 0
        self._create_entities()
        for p in self.defensive_policies:
            p.reset()
        self._run_target_assignment()
        self._update_lock_on_map()
        self.prev_dists_to_hvt = [
            off.distance_to(self.hvt.x, self.hvt.y, self.hvt.z)
            for off in self.offensives
        ]
        self.initial_dists_to_hvt = [max(d, 1.0) for d in self.prev_dists_to_hvt]
        self.prev_team_min_dist = min(self.prev_dists_to_hvt)
        self.hit_count = 0
        self.hit_indices = []
        self.kill_events = []
        self.escape_events_total = []
        self.engagement_tracking = {}
        self.miss_cooldowns = {}
        self.lock_events_log = []

        # ====== V28: Analytic Priors per-episode init ======
        if self._ap_enabled:
            self._ap_prev_q_matrix = None
            if self.ap_config.get("enable_assignment_mismatch_reward", False):
                self._ap_fixed_assignment = compute_initial_assignment(
                    self.offensives, self.defensives,
                    self.config["fov_half_angle"], self.ap_config)
            else:
                self._ap_fixed_assignment = {}
            self._ap_prev_M_tilde = None
            self._ap_prev_hvt_omega = np.zeros(self.n_offensive, dtype=np.float32)
            self._ap_prev_Phi_decoy = None
            self._ap_prev_N_eff = None

        # V28: Reset analytic data caches
        self._ap_decoy_info = {}
        self._ap_pen_info = {}
        self._ap_esc_info = {}
        self._ap_hvt_info = {}
        self._ap_Z_matrix = np.zeros((self.n_offensive, self.n_defensive), dtype=np.float32)
        self._ap_Z_tilde = np.zeros(self.n_offensive, dtype=np.float32)
        self._ap_psi_agg = np.zeros(self.n_offensive, dtype=np.float32)
        self._ap_Gamma_matrix = [[0.0] * self.n_defensive for _ in range(self.n_offensive)]
        self._ap_Xi_matrix = [[0.0] * self.n_defensive for _ in range(self.n_offensive)]
        self._ap_prev_E_esc = None

        # V36fix: Compute initial _ap_hvt_info so first obs is not all-zeros
        if self._ap_enabled:
            ap = self.config.get("analytic_priors", {})
            pn_nav_gain = ap.get("pn_nav_gain", 3.0)
            rho_list, closing_list, omega_list = [], [], []
            omega_dot_list, pn_hint_list, P_hit_list = [], [], []
            for i, off in enumerate(self.offensives):
                if not off.alive or off.hit_hvt:
                    rho_list.append(0.0); closing_list.append(0.0)
                    omega_list.append(0.0); omega_dot_list.append(0.0)
                    pn_hint_list.append(0.0); P_hit_list.append(0.0)
                    continue
                feats = compute_hvt_guidance_features(
                    off, self.hvt, self.dt,
                    prev_omega_los=self._ap_prev_hvt_omega[i],
                    pn_nav_gain=pn_nav_gain)
                self._ap_prev_hvt_omega[i] = feats["omega_los"]
                rho_list.append(feats["rho"])
                closing_list.append(feats["closing_speed"])
                omega_list.append(feats["omega_los"])
                omega_dot_list.append(feats["omega_los_dot"])
                pn_hint_list.append(feats["pn_hint"])
                P_hit_list.append(0.0)  # P_hit starts at 0 (far from target)
            self._ap_hvt_info = {
                "rho_per_agent": rho_list,
                "closing_per_agent": closing_list,
                "omega_per_agent": omega_list,
                "omega_dot_per_agent": omega_dot_list,
                "pn_hint_per_agent": pn_hint_list,
                "P_hit_per_agent": P_hit_list,
            }

        obs = self._get_obs()
        share_obs = self._get_share_obs()
        avail = self._get_avail_actions()
        return obs, share_obs, avail

    def step(self, actions):
        cfg = self.config
        self.current_step += 1

        # 0. 保存上一步位置 (供CPA连续命中检测)
        for off in self.offensives:
            off._prev_pos = (off.x, off.y, off.z)
        # V23: 也保存防御方前序位置 (供拦截器CPA双杀检测)
        for d in self.defensives:
            d._prev_pos = (d.x, d.y, d.z)

        # 1. 进攻方执行动作
        for i, off in enumerate(self.offensives):
            if off.alive and not off.hit_hvt:
                action = np.array(actions[i], dtype=np.float32)
                off.step_with_action(action, self.dt)

        # 2. 防御方执行 3D PN
        for i, policy in enumerate(self.defensive_policies):
            d = self.defensives[i]
            if d.alive:
                ax_cmd, an_pitch_cmd, an_yaw_cmd = policy.get_action(self.offensives, self.dt)
                d.step(ax_cmd, an_pitch_cmd, an_yaw_cmd, self.dt)

        # 3. 更新探测状态
        self._update_detection()

        # 4. ★ V11_01: 击杀、逃逸与命中 — 核心新逻辑 ★
        alive_before_off = [off.alive for off in self.offensives]
        alive_before_def = [d.alive for d in self.defensives]
        step_hits, step_escapes, step_misses = self._check_kills_escapes_and_hits()

        # 5. 边界与撞地
        self._check_boundary_and_ground()

        just_killed_off = [alive_before_off[i] and not self.offensives[i].alive
                           for i in range(self.n_offensive)]
        just_killed_def = [alive_before_def[i] and not self.defensives[i].alive
                           for i in range(self.n_defensive)]

        # 6. V22: 每步更新拦截器锁定状态 (FOV触发锁定)
        #    V38: 排他锁定 — 已被锁定的进攻方不允许被其他拦截器重复锁定
        step_lock_events = []
        already_locked_offensives = set()
        # 先收集当前已被锁定的进攻方
        for di, policy in enumerate(self.defensive_policies):
            if (policy.lock_mode == InterceptorPolicy.STATE_LOCKED
                    and policy.current_locked_target_idx is not None):
                already_locked_offensives.add(policy.current_locked_target_idx)
        for di, policy in enumerate(self.defensive_policies):
            if self.defensives[di].alive:
                lock_ev = policy.update_lock_state(
                    self.offensives, self.current_step,
                    already_locked_offensives=already_locked_offensives)
                if lock_ev is not None:
                    # 新锁定成功，加入已锁定集合
                    if lock_ev.get("type") == "fov_trigger_lock":
                        already_locked_offensives.add(lock_ev["off_idx"])
                    step_lock_events.append(lock_ev)
                    self.lock_events_log.append(lock_ev)

        # 7. 更新锁定关系图 (基于FOV触发锁定状态)
        self._update_lock_on_map()

        # V22: 记录每步脱靶量
        point_target_cfg = self.config.get("point_target", {})
        if point_target_cfg.get("record_miss_distance", True):
            for off in self.offensives:
                if off.alive and not off.hit_hvt:
                    d_hvt = off.distance_to(self.hvt.x, self.hvt.y, self.hvt.z)
                    off.miss_distance_history.append(d_hvt)
                    if d_hvt < off.min_miss_distance:
                        off.min_miss_distance = d_hvt

        # 8. (V24: miss冷却已移除)

        # 9. V28: 计算 analytic priors (供 reward 和 obs 使用)
        ap_data = {}  # 传给 compute_rewards 的数据
        ap_info = {}  # 日志信息
        if self._ap_enabled:
            ap = self.ap_config
            self._ap_global_step += 1

            # --- Module 1: Cone Cost → Z_matrix, Z_tilde, psi_agg ---
            if ap.get("enable_cone_cost", False) and self._y_cache is not None:
                cone_cost_val, cone_info, q_matrix = compute_group_cone_cost(
                    self.offensives, self.defensives, cfg, ap,
                    self._y_cache, self.current_step,
                    prev_q_matrix=self._ap_prev_q_matrix)
                self._ap_prev_q_matrix = q_matrix
                Z_matrix = cone_info.pop("_Z_matrix", None)
                if Z_matrix is not None:
                    self._ap_Z_matrix = Z_matrix
                Z_tilde_arr = cone_info.get("_Z_tilde", np.zeros(self.n_offensive))
                psi_per_agent = cone_info.get("psi_agg_per_agent", [0.0] * self.n_offensive)
                for i in range(self.n_offensive):
                    self._ap_Z_tilde[i] = Z_tilde_arr[i] if i < len(Z_tilde_arr) else 0.0
                    self._ap_psi_agg[i] = psi_per_agent[i] if i < len(psi_per_agent) else 0.0
                ap_data["cone_cost_per_agent"] = list(self._ap_psi_agg)
                ap_data["Z_tilde_per_agent"] = list(self._ap_Z_tilde)
                ap_info.update(cone_info)
            else:
                ap_data["cone_cost_per_agent"] = [0.0] * self.n_offensive
                ap_data["Z_tilde_per_agent"] = [0.0] * self.n_offensive

            # --- Module 2: Decoy Game → U_decoy, role probs ---
            if ap.get("enable_decoy_game", False):
                # V29fix3: Save old Phi BEFORE computing new one
                old_Phi_decoy = self._ap_prev_Phi_decoy
                decoy_reward, per_agent_decoy, Phi_decoy, decoy_info = compute_decoy_game(
                    self.offensives, self.defensives, self.hvt, cfg, ap,
                    locked_by_map=self.locked_by_map,
                    prev_Phi_decoy=old_Phi_decoy)
                # Now update cache to new value
                self._ap_prev_Phi_decoy = Phi_decoy
                self._ap_decoy_info = decoy_info
                ap_data["U_decoy_per_agent"] = decoy_info.get("U_decoy_per_agent", [0.0] * self.n_offensive)
                ap_data["Phi_decoy"] = Phi_decoy
                ap_data["prev_Phi_decoy"] = old_Phi_decoy if old_Phi_decoy is not None else Phi_decoy
                ap_info.update(decoy_info)
            else:
                self._ap_decoy_info = {}
                ap_data["U_decoy_per_agent"] = [0.0] * self.n_offensive

            # --- Module 3a: LOS Escape → E_i_esc, Gamma, Xi matrices ---
            if ap.get("enable_escape_reward", False):
                escape_reward_val, per_agent_esc, escape_info = compute_escape_reward(
                    self.offensives, self.defensives, cfg, ap)
                self._ap_esc_info = escape_info
                E_esc = escape_info.get("E_i_esc", [0.0] * self.n_offensive)
                ap_data["E_esc_per_agent"] = E_esc
                # Store prev for progress reward
                ap_data["prev_E_esc_per_agent"] = self._ap_prev_E_esc
                self._ap_prev_E_esc = list(E_esc)
                # 存储 Gamma/Xi 矩阵 (per-pair)
                Gamma_mat = escape_info.get("_Gamma_matrix", None)
                Xi_mat = escape_info.get("_Xi_matrix", None)
                if Gamma_mat is not None:
                    self._ap_Gamma_matrix = Gamma_mat
                if Xi_mat is not None:
                    self._ap_Xi_matrix = Xi_mat
                ap_info.update(escape_info)
            else:
                self._ap_esc_info = {"E_i_esc": [0.0] * self.n_offensive}
                ap_data["E_esc_per_agent"] = [0.0] * self.n_offensive

            # --- Module 3b: Effective Penetration → P_pen, N_eff ---
            if ap.get("enable_effective_penetration", False):
                _cone_risk = list(self._ap_psi_agg)
                _lock_pressure = (self._ap_decoy_info.get(
                    "lock_pressure_per_agent", [0.0] * self.n_offensive))
                _locked_by_count = [len(self.locked_by_map.get(i, []))
                                    for i in range(self.n_offensive)]
                _E_i_esc = ap_data.get("E_esc_per_agent", [0.0] * self.n_offensive)

                pen_reward, per_agent_pen, N_eff, pen_info = compute_effective_penetration(
                    self.offensives, self.defensives, self.hvt, cfg, ap,
                    cone_risk_per_agent=_cone_risk,
                    lock_pressure_per_agent=_lock_pressure,
                    locked_by_count_per_agent=_locked_by_count,
                    E_i_esc_per_agent=_E_i_esc,
                    prev_N_eff=self._ap_prev_N_eff)
                self._ap_prev_N_eff = N_eff
                self._ap_pen_info = pen_info
                ap_data["P_pen_per_agent"] = pen_info.get("P_pen_per_agent", [0.0] * self.n_offensive)
                ap_data["N_eff"] = N_eff
                ap_info.update(pen_info)
            else:
                self._ap_pen_info = {}
                ap_data["P_pen_per_agent"] = [0.0] * self.n_offensive

            # --- Module 4: HVT Guidance → rho, closing, omega, omega_dot, pn_hint, P_hit ---
            if ap.get("enable_hvt_guidance", False):
                pn_nav_gain = ap.get("pn_nav_gain", 3.0)
                rho_list = []
                closing_list = []
                omega_list = []
                omega_dot_list = []
                pn_hint_list = []
                P_hit_list = []
                for i, off in enumerate(self.offensives):
                    if not off.alive or off.hit_hvt:
                        rho_list.append(0.0)
                        closing_list.append(0.0)
                        omega_list.append(0.0)
                        omega_dot_list.append(0.0)
                        pn_hint_list.append(0.0)
                        P_hit_list.append(0.0)
                        continue
                    feats = compute_hvt_guidance_features(
                        off, self.hvt, self.dt,
                        prev_omega_los=self._ap_prev_hvt_omega[i],
                        pn_nav_gain=pn_nav_gain)
                    self._ap_prev_hvt_omega[i] = feats["omega_los"]
                    rho_list.append(feats["rho"])
                    closing_list.append(feats["closing_speed"])
                    omega_list.append(feats["omega_los"])
                    omega_dot_list.append(feats["omega_los_dot"])
                    pn_hint_list.append(feats["pn_hint"])

                    # V29fix2: P_i_hit = sigma(kappa_h*(d_soft - rho)) * sigma(kappa_c*Vc) * sigma(-kappa_w*|omega|)
                    # d_soft=10m (soft shaping), hard hit still 5m
                    kappa_h = ap.get("kappa_h", 2.0)
                    kappa_c = ap.get("kappa_c", 1.0)
                    kappa_w = ap.get("kappa_omega", 2.0)
                    d_soft = 10.0  # V29fix2: soft shaping radius, NOT rho/100
                    from .reward_cost import _sigmoid as _sig
                    p_hit = (_sig(kappa_h * (d_soft - feats["rho"]))
                             * _sig(kappa_c * feats["closing_speed"])
                             * _sig(-kappa_w * abs(feats["omega_los"])))
                    P_hit_list.append(float(p_hit))

                self._ap_hvt_info = {
                    "rho_per_agent": rho_list,
                    "closing_per_agent": closing_list,
                    "omega_per_agent": omega_list,
                    "omega_dot_per_agent": omega_dot_list,
                    "pn_hint_per_agent": pn_hint_list,
                    "P_hit_per_agent": P_hit_list,
                }
                ap_data["hvt_omega_los_per_agent"] = omega_list
                ap_data["P_hit_per_agent"] = P_hit_list
                self._ap_pen_info["P_hit_per_agent"] = P_hit_list
                ap_info.update({
                    "hvt_rho_mean": float(np.mean([r for r in rho_list if r > 0])) if any(r > 0 for r in rho_list) else 0.0,
                    "hvt_closing_speed_mean": float(np.mean([c for c in closing_list if c != 0])) if any(c != 0 for c in closing_list) else 0.0,
                })
            else:
                self._ap_hvt_info = {}
                ap_data["hvt_omega_los_per_agent"] = [0.0] * self.n_offensive
                ap_data["P_hit_per_agent"] = [0.0] * self.n_offensive

            # 传入锁定映射
            ap_data["locked_target_by_defender"] = dict(self.locked_target_by_defender)

        # 10. V28: 奖励计算 — 使用新的 compute_rewards
        rewards_list, reward_info = compute_rewards(
            self.offensives, self.defensives, self.hvt, cfg,
            self.prev_dists_to_hvt, hit_events=step_hits,
            current_step=self.current_step,
            just_killed=just_killed_off,
            just_killed_def=just_killed_def,
            lock_on_map=self.lock_on_map,
            prev_team_min_dist=self.prev_team_min_dist,
            escape_events=step_escapes,
            miss_events=step_misses,
            defensive_policies=self.defensive_policies,
            ap_data=ap_data,
            raw_actions=actions)

        # 更新prev状态
        cur_dists = []
        for off in self.offensives:
            if off.alive and not off.hit_hvt:
                cur_dists.append(off.distance_to(self.hvt.x, self.hvt.y, self.hvt.z))
        self.prev_team_min_dist = min(cur_dists) if cur_dists else self.prev_team_min_dist

        self.prev_dists_to_hvt = [
            off.distance_to(self.hvt.x, self.hvt.y, self.hvt.z)
            if off.alive and not off.hit_hvt else float('inf')
            for off in self.offensives
        ]

        # 10.5 V28: 成本已并入 reward, compute_costs 返回全零
        costs_list, cost_info = compute_costs(self.offensives, self.defensives, cfg)

        # 11. 终止
        done, done_reason = self._check_done()

        # V28: 终端奖励
        if done:
            terminal_rewards, terminal_info = compute_terminal_rewards(
                self.offensives, self.hvt, cfg, ap_data=ap_data)
            for i in range(self.n_agents):
                rewards_list[i] += terminal_rewards[i]
            ap_info.update(terminal_info)

            # timeout 额外按距离惩罚
            if done_reason == "timeout":
                for i in range(self.n_agents):
                    rewards_list[i] += cfg["reward"]["timeout_penalty"]
                timeout_dist_coef = cfg["reward"].get("timeout_distance_penalty_coef", 0.0)
                if timeout_dist_coef != 0.0:
                    for i, off in enumerate(self.offensives):
                        if not off.alive or off.hit_hvt:
                            continue
                        dist_hvt = off.distance_to(self.hvt.x, self.hvt.y, self.hvt.z)
                        dist_ratio = np.clip(dist_hvt / max(cfg.get("obs_range", 2500.0), 1.0), 0.0, 2.0)
                        rewards_list[i] -= timeout_dist_coef * dist_ratio

        # 12. 组装
        rewards = [[r] for r in rewards_list]
        costs = [[c] for c in costs_list]
        dones = [done] * self.n_agents

        off_alive = sum(1 for o in self.offensives if o.alive)
        def_alive = sum(1 for d in self.defensives if d.alive)
        avg_exposure = np.mean([o.total_exposure_steps for o in self.offensives])
        first_det_steps = [o.first_detected_step for o in self.offensives if o.first_detected_step >= 0]
        first_det_avg = np.mean(first_det_steps) if first_det_steps else -1.0
        n_detected_now = sum(1 for o in self.offensives if o.alive and o.detected)
        # V24: 计算每个进攻方吸引了多少拦截器
        decoy_counts = []
        for oi, off in enumerate(self.offensives):
            n_chasers = sum(1 for dp in self.defensive_policies
                           if dp.interceptor.alive and dp.target is off)
            decoy_counts.append(n_chasers)

        two_stage = cfg["two_stage_eval"]
        if two_stage["enabled"]:
            stage1 = two_stage["stage1_weight"] * (1.0 if self.hit_count > 0 else 0.0)
            stage2 = two_stage["stage2_weight"] * self.hit_count
            two_stage_score = stage1 + stage2
        else:
            two_stage_score = float(self.hit_count > 0)

        info = {
            "cost": [[c] for c in costs_list],
            "success": self.hit_count > 0,
            "done_reason": done_reason,
            "hit_count": self.hit_count,
            "hit_indices": list(self.hit_indices),
            "two_stage_score": two_stage_score,
            "offensive_alive": off_alive,
            "defensive_alive": def_alive,
            "offensive_killed": self.n_offensive - off_alive,
            "avg_exposure_steps": avg_exposure,
            "first_detected_avg": first_det_avg,
            "n_detected_now": n_detected_now,
            "avg_exposure_rate": avg_exposure / max(self.current_step, 1),
            "kill_events": list(self.kill_events),
            "lock_on_map": dict(self.lock_on_map),
            # V22: 显式锁定映射
            "locked_target_by_defender": dict(self.locked_target_by_defender),
            "locked_by_map": dict(self.locked_by_map),
            "step_lock_events": step_lock_events,
            "n_locked_defenders": sum(
                1 for p in self.defensive_policies
                if p.lock_mode == InterceptorPolicy.STATE_LOCKED),
            # V22: 点目标脱靶量
            "terminal_miss_distance_per_agent": [
                off.min_miss_distance for off in self.offensives],
            "terminal_miss_distance_min": min(
                (off.min_miss_distance for off in self.offensives),
                default=float('inf')),
            # 诱饵统计 (V24)
            "decoy_counts_per_agent": decoy_counts,
            "n_escapes_total": 0,
            "n_escaped_agents": 0,
            "step_escapes": 0,
            "step_misses": 0,
        }
        # V28: Analytic priors info
        if ap_info:
            info.update(ap_info)
        if done:
            info["bad_transition"] = (done_reason == "timeout")
            # V22: 终端统计 (youneedread 6.7)
            info["num_hit_hvt"] = self.hit_count
            info["first_hit_time"] = (
                min(off.hit_time for off in self.offensives if off.hit_time >= 0)
                if any(off.hit_time >= 0 for off in self.offensives) else -1)
            info["hit_agent_ids"] = list(self.hit_indices)
            info["lock_events_log"] = list(self.lock_events_log)

        infos = [info for _ in range(self.n_agents)]
        obs = self._get_obs()
        share_obs = self._get_share_obs()
        avail = self._get_avail_actions()
        return obs, share_obs, rewards, costs, dones, infos, avail

    def _update_detection(self):
        cfg = self.config
        fov_half = cfg["fov_half_angle"]
        det_range = cfg["detection_range"]
        for off in self.offensives:
            if not off.alive:
                off.update_detection(False, 0, self.current_step)
                continue
            count = 0
            for d in self.defensives:
                if d.alive and d.is_in_fov(off.x, off.y, off.z, fov_half, det_range):
                    count += 1
            off.update_detection(count > 0, count, self.current_step)

    def _check_kills_escapes_and_hits(self):
        """
        V24 简化: 纯碰撞击杀 + HVT命中

        击杀逻辑:
          - CPA轨迹检测: 上步→本步线段最近距离 < collision_kill_range → 双杀
          - 离散碰撞: 当前距离 < collision_kill_range → 双杀

        命中HVT:
          - 离散: dist < hit_hvt_range
          - CPA连续: 线段最近点 < hit_hvt_range
        """
        cfg = self.config
        collision_kill_range = cfg.get("collision_kill_range", 5.0)

        step_hits = []
        step_escapes = []   # V24: 不再产生逃逸事件，保留接口兼容
        step_misses = []

        # ——————————————————————————————————
        # A. 拦截器 vs 进攻方: 碰撞击杀
        # ——————————————————————————————————
        for di, d in enumerate(self.defensives):
            if not d.alive:
                continue

            for oi, off in enumerate(self.offensives):
                if not off.alive or off.hit_hvt:
                    continue

                dist = d.distance_3d(off)

                # --- CPA轨迹检测 ---
                if hasattr(d, '_prev_pos') and hasattr(off, '_prev_pos'):
                    dx0, dy0, dz0 = d._prev_pos
                    dx1, dy1, dz1 = d.x, d.y, d.z
                    ox0, oy0, oz0 = off._prev_pos
                    ox1, oy1, oz1 = off.x, off.y, off.z
                    _cpa_hit = False
                    for _t in [i * 0.1 for i in range(11)]:
                        _pd = np.array([dx0 + _t*(dx1-dx0), dy0 + _t*(dy1-dy0), dz0 + _t*(dz1-dz0)])
                        _po = np.array([ox0 + _t*(ox1-ox0), oy0 + _t*(oy1-oy0), oz0 + _t*(oz1-oz0)])
                        if np.linalg.norm(_pd - _po) < collision_kill_range:
                            _cpa_hit = True
                            break
                    if _cpa_hit and off.alive and d.alive:
                        off.kill()
                        d.kill()
                        self.kill_events.append({
                            "step": self.current_step,
                            "defensive_id": d.uid,
                            "offensive_id": off.uid,
                            "mutual_kill": True,
                            "type": "cpa_kill",
                            "dist": dist,
                        })
                        break  # 该防御方已死

                # --- 离散碰撞击杀 ---
                if not d.alive:
                    break
                if dist < collision_kill_range:
                    off.kill()
                    d.kill()
                    self.kill_events.append({
                        "step": self.current_step,
                        "defensive_id": d.uid,
                        "offensive_id": off.uid,
                        "mutual_kill": True,
                        "type": "collision",
                    })
                    break  # 该防御方已死

        # ——————————————————————————————————
        # B. 命中HVT判定 (V22fix: CPA连续检测 + 离散检测)
        # ——————————————————————————————————
        hvt_range = cfg.get("point_target", {}).get("hit_threshold", 5.0)  # V29fix1: 唯一权威字段
        hx, hy, hz = self.hvt.x, self.hvt.y, self.hvt.z
        for i, off in enumerate(self.offensives):
            if not off.alive or off.hit_hvt:
                continue
            # 离散位置检测
            dist_now = off.distance_to(hx, hy, hz)
            if dist_now < hvt_range:
                off.mark_hit_hvt()
                off.hit_time = self.current_step
                self.hit_count += 1
                self.hit_indices.append(i)
                step_hits.append(i)
                continue
            # CPA连续检测: 上一步位置→当前位置线段上最近点
            if hasattr(off, '_prev_pos'):
                px, py, pz = off._prev_pos
                dx, dy, dz = off.x - px, off.y - py, off.z - pz
                seg_len_sq = dx*dx + dy*dy + dz*dz
                if seg_len_sq > 1e-6:
                    t = max(0.0, min(1.0, (
                        (hx - px)*dx + (hy - py)*dy + (hz - pz)*dz
                    ) / seg_len_sq))
                    cx = px + t*dx
                    cy = py + t*dy
                    cz = pz + t*dz
                    cpa_dist = np.sqrt((cx-hx)**2 + (cy-hy)**2 + (cz-hz)**2)
                    if cpa_dist < hvt_range:
                        off.mark_hit_hvt()
                        off.hit_time = self.current_step
                        self.hit_count += 1
                        self.hit_indices.append(i)
                        step_hits.append(i)

        return step_hits, step_escapes, step_misses

    def _update_miss_cooldowns(self):
        """递减miss冷却计数器"""
        expired = []
        for key in self.miss_cooldowns:
            self.miss_cooldowns[key] -= 1
            if self.miss_cooldowns[key] <= 0:
                expired.append(key)
        for key in expired:
            del self.miss_cooldowns[key]

    def _check_boundary_and_ground(self):
        cfg = self.config
        map_size = cfg["map_size"]
        z_min = cfg["z_min"]
        for off in self.offensives:
            if off.alive:
                if abs(off.x) > map_size * 1.5 or abs(off.y) > map_size * 1.5:
                    off.kill()
                if off.z < z_min:
                    off.kill()

    def _check_done(self):
        if self.hit_count > 0:
            return True, "success"
        any_alive = any(o.alive and not o.hit_hvt for o in self.offensives)
        if not any_alive:
            return True, "all_killed"
        if self.current_step >= self.max_steps:
            return True, "timeout"
        return False, ""

    def _get_obs(self):
        """
        V41 观测空间 — 理论驱动 (Theory-Grounded), 总维度 22.

        设计原则 (来自 newswarm.tex):
          1. 不使用任何绝对位置 — 全部相对量、视线量、闭合速度等理论物理量
          2. HVT 不给相对位置, 只给 LOS 角速度+闭合速度 → 引导 agent 学 PN 制导
          3. 拦截器侧用 tex 中的 q_ij (FOV锥裕度), Γ_ij (跟踪失配裕度) 等理论核心量
          4. 群体先验 P_pen_i, P_hit_i 作为辅助观测 (tex 4.1 节)

        分块:
          1. Self kinematic (5):  v, sin(gamma), cos(gamma), an_pitch, an_yaw
          2. HVT guidance (3):    omega_LOS_pitch_iH, omega_LOS_yaw_iH, V_c_iH
                                  (tex P_hit 公式中的 omega 与 V_c, ρ 不给以避免靠位置硬背)
          3. Top-K threats (5*2): per interceptor:
              q_ij        — FOV 锥裕度 (tex 公式 q_ij)
              V_c_ij      — 闭合速度
              |ω_LOS_ij|  — 视线角速度模长 (tex 局部逃逸机理核心)
              Γ_ij        — 跟踪失配裕度 = |ω_LOS|·V_j / a_j_max - 1 (归一化)
              is_locking_me — 是否锁定我 (来自拦截器状态机)
          4. Team & priors (3):   n_mates_drawing_fire / (N-1), P_pen_i, P_hit_i
          5. Time (1):            t/T
        """
        cfg = self.config
        vel_range = max(cfg["vel_range"], 1.0)
        fov_half = cfg["fov_half_angle"]

        hvt_x, hvt_y, hvt_z = self.hvt.x, self.hvt.y, self.hvt.z

        # 预计算
        pen_info = self._ap_pen_info
        hvt_info = self._ap_hvt_info

        # 拦截器最大法向加速度 (用于 Gamma_ij)
        a_def_max = cfg["defensive"].get("an_pitch_max", 5.0 * 9.81)

        omega_norm = 0.5    # 角速度归一化 (rad/s)
        an_norm = 25.0      # 加速度归一化 (m/s^2)

        obs_list = []
        for ai, agent in enumerate(self.offensives):
            obs = []

            # ============================================
            # 1. Self kinematic (5) — 仅自身飞行状态, 无任何绝对位置
            # ============================================
            obs.extend([
                agent.v / vel_range,
                np.sin(agent.gamma),
                np.cos(agent.gamma),
                np.clip(agent.an_pitch / an_norm, -1.0, 1.0),
                np.clip(agent.an_yaw / an_norm, -1.0, 1.0),
            ])

            # ============================================
            # 2. HVT guidance (3) — 仅 LOS 角速度 (pitch/yaw 分量) + 闭合速度
            #    引导 agent 学会 PN 制导: 减小 LOS rate 即可命中
            # ============================================
            if agent.alive and not agent.hit_hvt:
                v_off = self._vel3d(agent)
                r_iH = np.array([hvt_x - agent.x, hvt_y - agent.y, hvt_z - agent.z])
                rho_iH = float(np.linalg.norm(r_iH))
                rho_safe = max(rho_iH, 1.0)
                # 闭合速度 V_c = -dρ/dt = (r·v)/ρ  (HVT 静止)
                vc_iH = float(np.dot(r_iH, v_off) / rho_safe)
                # LOS 角速度 (方位角速率 + 俯仰角速率), 用解析公式从 r, v 直接算
                #   λ_az = atan2(r_y, r_x)        ⇒  dλ_az/dt = (r_x v_y - r_y v_x) / (r_x² + r_y²)
                #   λ_el = asin(r_z / ρ)          ⇒  dλ_el/dt = (v_z * ρ_h² - r_z * (r_x v_x + r_y v_y))
                #                                                / (ρ² * ρ_h)
                rx, ry, rz = r_iH
                vx, vy, vz = v_off
                rho_h_sq = rx * rx + ry * ry
                rho_h = float(np.sqrt(max(rho_h_sq, 1e-6)))
                d_az = float((rx * vy - ry * vx) / max(rho_h_sq, 1e-6))
                d_el = float((vz * rho_h_sq - rz * (rx * vx + ry * vy))
                             / (max(rho_iH * rho_iH, 1e-6) * rho_h))
                # 机头朝向与 LOS 单位向量的对准 cos —— 纯方向, 不含距离
                v_norm = float(np.linalg.norm(v_off))
                if v_norm > 1e-3:
                    align_cos = float(np.dot(v_off, r_iH) / (v_norm * rho_safe))
                else:
                    align_cos = 0.0
            else:
                vc_iH = 0.0
                d_az = 0.0
                d_el = 0.0
                align_cos = 0.0

            obs.extend([
                np.clip(d_az / omega_norm, -1.0, 1.0),
                np.clip(d_el / omega_norm, -1.0, 1.0),
                np.clip(vc_iH / vel_range, -1.0, 1.0),
                np.clip(align_cos, -1.0, 1.0),
            ])

            # ============================================
            # 3. Top-K threats (5*K=10) — 全部为 tex 中的理论核心量
            # ============================================
            threat_scores = self._compute_threat_scores(ai, agent)
            threat_scores.sort(key=lambda x: -x[0])

            for k in range(self._n_top_threats):
                if k < len(threat_scores) and threat_scores[k][0] > -100:
                    _, di = threat_scores[k]
                    d = self.defensives[di]
                    if d.alive and agent.alive:
                        v_off = self._vel3d(agent)
                        v_def = self._vel3d(d)
                        r_ij = np.array([agent.x - d.x, agent.y - d.y, agent.z - d.z])
                        rho_ij = float(np.linalg.norm(r_ij))
                        rho_safe = max(rho_ij, 1e-3)
                        v_rel = v_off - v_def
                        # 闭合速度 V_c_ij = -d(ρ)/dt = -(r·v_rel)/ρ
                        vc_ij = float(-np.dot(r_ij, v_rel) / rho_safe)
                        # FOV 锥裕度 q_ij (tex 公式)
                        cg = np.cos(d.gamma)
                        b_j = np.array([cg * np.cos(d.heading),
                                        cg * np.sin(d.heading),
                                        np.sin(d.gamma)])
                        cos_theta = np.clip(np.dot(b_j, r_ij) / rho_safe, -1.0, 1.0)
                        q_ij = cos_theta - np.cos(fov_half)
                        # 视线角速度模长 (tex 公式)
                        omega_los_ij = float(np.linalg.norm(np.cross(r_ij, v_rel))
                                             / max(rho_safe * rho_safe, 1e-6))
                        # 跟踪失配裕度 Γ_ij = |ω| - a_max/V_j (tex 公式)
                        # 归一化为 (|ω| * V_j / a_max - 1) ∈ [-1, +∞), 取 tanh 限幅
                        v_def_safe = max(d.v, 1.0)
                        gamma_ratio = omega_los_ij * v_def_safe / max(a_def_max, 1.0) - 1.0
                        Gamma_norm = float(np.tanh(gamma_ratio))
                        # 是否锁定我
                        policy = self.defensive_policies[di]
                        is_locking = 1.0 if (policy.lock_mode == InterceptorPolicy.STATE_LOCKED
                                             and policy.current_locked_target_idx == ai) else 0.0

                        obs.extend([
                            np.clip(q_ij, -1.0, 1.0),
                            np.clip(vc_ij / vel_range, -1.0, 1.0),
                            np.clip(omega_los_ij / omega_norm, 0.0, 2.0),
                            Gamma_norm,
                            is_locking,
                        ])
                    else:
                        obs.extend([0.0] * self._dim_per_threat)
                else:
                    obs.extend([0.0] * self._dim_per_threat)

            # ============================================
            # 4. Team & priors (3) — 注意力转移信号 + 理论先验
            # ============================================
            n_mates_drawing_fire = 0
            for k, other in enumerate(self.offensives):
                if k == ai or not other.alive or other.hit_hvt:
                    continue
                if other.locked_by_count > 0:
                    n_mates_drawing_fire += 1

            P_pen_i = pen_info.get("P_pen_per_agent", [0.0] * self.n_offensive)[ai]
            P_hit_i = pen_info.get("P_hit_per_agent", [0.0] * self.n_offensive)[ai]

            obs.extend([
                n_mates_drawing_fire / max(self.n_offensive - 1, 1),
                np.clip(P_pen_i, 0.0, 1.0),
                np.clip(P_hit_i, 0.0, 1.0),
            ])

            # ============================================
            # 5. Time (1)
            # ============================================
            obs.append(self.current_step / self.max_steps)

            obs_list.append(np.array(obs, dtype=np.float32))
        return obs_list

    @staticmethod
    def _vel3d(entity):
        """3D velocity vector"""
        cg = np.cos(entity.gamma)
        return np.array([entity.v * cg * np.cos(entity.heading),
                         entity.v * cg * np.sin(entity.heading),
                         entity.v * np.sin(entity.gamma)])

    def _compute_threat_scores(self, ai, agent):
        """计算每个拦截器对进攻方 ai 的威胁分数, 用于选 Top-2"""
        if not agent.alive:
            return []
        lambda_rho = 1.0
        lambda_q = 1.0
        lambda_L = 2.0
        fov_half = self.config["fov_half_angle"]
        scores = []
        for di, d in enumerate(self.defensives):
            if not d.alive:
                scores.append((-1000.0, di))
                continue
            rho = agent.distance_3d(d)
            # (1) 距离越近越危险
            dist_score = lambda_rho / (rho + 1.0)
            # (2) 在 FOV 内更危险
            r_ij = np.array([agent.x - d.x, agent.y - d.y, agent.z - d.z])
            rho_safe = max(rho, 1e-3)
            cg = np.cos(d.gamma)
            b_j = np.array([cg * np.cos(d.heading), cg * np.sin(d.heading), np.sin(d.gamma)])
            q_ij = np.dot(b_j, r_ij) / rho_safe - np.cos(fov_half)
            fov_score = lambda_q * max(q_ij, 0.0)
            # (3) 锁定我 → 高危
            policy = self.defensive_policies[di]
            lock_score = lambda_L if (policy.lock_mode == InterceptorPolicy.STATE_LOCKED
                                      and policy.current_locked_target_idx == ai) else 0.0
            total = dist_score + fov_score + lock_score
            scores.append((total, di))
        return scores

    def _get_share_obs(self):
        """V28 共享观测: 全局完整状态"""
        cfg = self.config
        obs_range = max(cfg["obs_range"], 1.0)
        vel_range = max(cfg["vel_range"], 1.0)
        z_range = max(cfg.get("z_range", 1000.0), 1.0)
        so = []
        for off in self.offensives:
            so.extend([
                off.x / obs_range, off.y / obs_range, off.z / z_range,
                off.v / vel_range, off.heading / np.pi,
                off.gamma / (np.pi / 4),
                off.ax / 20.0, off.an_pitch / 25.0, off.an_yaw / 25.0,
                float(off.alive),
            ])
        for d in self.defensives:
            so.extend([
                d.x / obs_range, d.y / obs_range, d.z / z_range,
                d.v / vel_range, d.heading / np.pi,
                d.gamma / (np.pi / 4),
                float(d.alive),
            ])
        so.extend([self.hvt.x / obs_range, self.hvt.y / obs_range,
                    self.hvt.z / z_range])
        alive_dists = [off.distance_to(self.hvt.x, self.hvt.y, self.hvt.z) / obs_range
                       for off in self.offensives if off.alive]
        min_dist = min(alive_dists) if alive_dists else 1.0
        n_det = sum(1 for o in self.offensives if o.alive and o.detected)
        off_alive_ratio = sum(1 for o in self.offensives if o.alive) / max(self.n_offensive, 1)
        def_alive_ratio = sum(1 for d in self.defensives if d.alive) / max(self.n_defensive, 1)
        so.extend([min_dist, n_det / max(self.n_offensive, 1),
                    off_alive_ratio, def_alive_ratio,
                    self.current_step / self.max_steps])
        # team penetration score
        pen_info = self._ap_pen_info
        team_pen = pen_info.get("N_eff", 0.0)
        so.append(float(np.clip(team_pen / max(self.n_offensive, 1), 0.0, 1.0)))
        so_array = np.array(so, dtype=np.float32)
        return [so_array.copy() for _ in range(self.n_agents)]

    def _get_avail_actions(self):
        return np.ones((self.n_agents, 3), dtype=np.float32)

    def close(self):
        pass

    def render(self, mode="human"):
        pass

    def get_env_info(self):
        return {
            "n_agents": self.n_agents,
            "n_offensive": self.n_offensive,
            "n_defensive": self.n_defensive,
            "obs_dim": self.obs_dim,
            "share_obs_dim": self.share_obs_dim,
            "action_dim": 3,
            "map_size": self.config["map_size"],
        }
