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
from .dynamics import action_to_overload_3d
from .reward_cost import compute_rewards, compute_costs
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
        self.hit_count = 0
        self.hit_indices = []
        self.kill_events = []
        self.escape_events_total = []      # V11_01: 全局逃逸记录
        self.assignments = {}
        self.lock_on_map = {}
        self.prev_team_min_dist = None

        # V11_01: 交战跟踪
        self.engagement_tracking = {}      # {(def_idx, off_idx): tracking_steps}
        self.miss_cooldowns = {}           # {(def_idx, off_idx): remaining_cooldown}

        # ====== Analytic Priors State ======
        self.ap_config = cfg.get("analytic_priors", {})
        self._ap_enabled = (self.ap_config.get("enable_cone_cost", False)
                            or self.ap_config.get("enable_assignment_mismatch_reward", False)
                            or self.ap_config.get("enable_escape_reward", False)
                            or self.ap_config.get("enable_hvt_guidance", False))
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

        # Per-episode analytic priors state (initialized in reset)
        self._ap_prev_q_matrix = None
        self._ap_fixed_assignment = {}
        self._ap_prev_M_tilde = None
        self._ap_episode_max_M_tilde = 0.0
        self._ap_cone_escape_success = 0  # escape while cone risk was high
        # Per-agent analytic obs cache (for _get_obs)
        self._ap_obs_cache = np.zeros((self.n_offensive, self._ap_obs_dim), dtype=np.float32)
        self._ap_prev_hvt_omega = np.zeros(self.n_offensive, dtype=np.float32)
        self._ap_penetration_score = np.zeros(self.n_offensive, dtype=np.float32)
        self._ap_team_penetration_score = 0.0
        self._ap_attack_gate_reward = np.zeros(self.n_offensive, dtype=np.float32)
        # Curriculum: global step counter (updated externally via step)
        self._ap_global_step = 0

    def _compute_space_dims(self):
        cfg = self.config
        n_def = cfg["n_defensive"]
        n_off = cfg["n_offensive"]
        self.obs_k_def = min(n_def, 4)
        # V11_01 观测空间 — 增加交战态势信息
        # self: 9, hvt: 3
        # 防御方(最近K个): 每个11维 (增加2维: overload_saturation, engagement_state)
        # 队友: 每个10维 (增加1维: escaped状态)
        # 暴露: 3, 全局: 2
        # 协同态势: 5 (增加1维: 已逃脱拦截器数)
        self._base_obs_dim = 9 + 3 + 11 * self.obs_k_def + 10 * (n_off - 1) + 3 + 2 + 5
        # Enhancement: analytic priors obs
        # base 4 dims: Z_tilde_i, psi_agg_i, M_tilde_norm, Xi_max_i
        # optional +6 dims: rho_hvt, closing_hvt, omega_los_hvt, omega_los_dot_hvt, pn_hint_hvt, penetration_score
        ap_cfg = cfg.get("analytic_priors", {})
        self._ap_base_obs_dim = 4 if ap_cfg.get("expose_analytic_obs", False) else 0
        self._ap_guidance_obs_dim = 6 if (ap_cfg.get("expose_analytic_obs", False)
                          and ap_cfg.get("expose_hvt_guidance_obs", False)) else 0
        self._ap_obs_dim = self._ap_base_obs_dim + self._ap_guidance_obs_dim
        self.obs_dim = self._base_obs_dim + self._ap_obs_dim
        self._ap_share_extra_dim = 1 if (ap_cfg.get("expose_penetration_share_obs", False)
                         and ap_cfg.get("enable_hvt_guidance", False)) else 0
        self.share_obs_dim = 10 * n_off + 7 * n_def + 3 + 5 + self._ap_share_extra_dim

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
            # V11_01: 逃逸标记
            off._escaped_interceptor = False
            off._n_escapes = 0
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
        self.assignments, _, _ = assign_targets(
            self.defensives, self.offensives, self.hvt, self.config)
        for def_idx, off_idx in self.assignments.items():
            if def_idx < len(self.defensive_policies):
                self.defensive_policies[def_idx].set_target(
                    off_idx, self.offensives[off_idx])

    def _update_lock_on_map(self):
        self.lock_on_map = {i: [] for i in range(self.n_offensive)}
        for def_idx, policy in enumerate(self.defensive_policies):
            d = self.defensives[def_idx]
            if not d.alive:
                continue
            if policy.target is not None and policy.target.alive:
                off_idx = policy.assigned_target_idx
                if off_idx is not None and off_idx < self.n_offensive:
                    self.lock_on_map[off_idx].append(def_idx)

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
        self.prev_team_min_dist = min(self.prev_dists_to_hvt)
        self.hit_count = 0
        self.hit_indices = []
        self.kill_events = []
        self.escape_events_total = []
        self.engagement_tracking = {}
        self.miss_cooldowns = {}

        # ====== Analytic Priors: per-episode init ======
        if self._ap_enabled:
            # Cone cost: initialize q_matrix to None (first step will use current as prev)
            self._ap_prev_q_matrix = None
            # Assignment mismatch: compute fixed assignment pi_0
            if self.ap_config.get("enable_assignment_mismatch_reward", False):
                self._ap_fixed_assignment = compute_initial_assignment(
                    self.offensives, self.defensives,
                    self.config["fov_half_angle"], self.ap_config)
            else:
                self._ap_fixed_assignment = {}
            self._ap_prev_M_tilde = None
            self._ap_episode_max_M_tilde = 0.0
            self._ap_cone_escape_success = 0
            self._ap_obs_cache = np.zeros((self.n_offensive, self._ap_obs_dim), dtype=np.float32)
            self._ap_prev_hvt_omega = np.zeros(self.n_offensive, dtype=np.float32)
            self._ap_penetration_score = np.zeros(self.n_offensive, dtype=np.float32)
            self._ap_team_penetration_score = 0.0
            self._ap_attack_gate_reward = np.zeros(self.n_offensive, dtype=np.float32)

        obs = self._get_obs()
        share_obs = self._get_share_obs()
        avail = self._get_avail_actions()
        return obs, share_obs, avail

    def step(self, actions):
        cfg = self.config
        self.current_step += 1

        # 1. 进攻方执行动作
        for i, off in enumerate(self.offensives):
            if off.alive and not off.hit_hvt:
                action = np.array(actions[i], dtype=np.float32)
                off.step_with_action(action, self.dt)

        # 2. 防御方执行 3D PN
        for i, policy in enumerate(self.defensive_policies):
            d = self.defensives[i]
            if d.alive:
                nx, ny, nz = policy.get_action(self.offensives, self.dt)
                d.step(nx, ny, nz, self.dt)

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

        # 6. 更新锁定关系图
        self._update_lock_on_map()

        # 7. 可选重分配
        if (cfg["assignment"]["reassign"] and
                self.current_step % cfg["assignment"]["reassign_interval"] == 0):
            self._run_target_assignment()
            self._update_lock_on_map()

        # 8. 更新miss冷却
        self._update_miss_cooldowns()

        # 9. 奖励 — 传入逃逸事件
        rewards_list, reward_info = compute_rewards(
            self.offensives, self.defensives, self.hvt, cfg,
            self.prev_dists_to_hvt, hit_events=step_hits,
            current_step=self.current_step,
            just_killed=just_killed_off,
            just_killed_def=just_killed_def,
            lock_on_map=self.lock_on_map,
            prev_team_min_dist=self.prev_team_min_dist,
            escape_events=step_escapes,
            miss_events=step_misses)

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

        # 10. 成本
        costs_list, cost_info = compute_costs(self.offensives, self.defensives, cfg)

        # ====== 10.5 Analytic Priors: cone cost / mismatch reward / escape reward ======
        ap_info = {}
        if self._ap_enabled:
            ap = self.ap_config
            self._ap_global_step += 1

            # --- Curriculum weight multiplier (only for reward shaping, NOT safety cost) ---
            cur_mult = 1.0
            if ap.get("curriculum_enabled", False):
                total = ap.get("curriculum_total_steps", 10000000)
                warmup = ap.get("curriculum_warmup_frac", 0.15)
                warmup_steps = total * warmup
                cur_mult = min(1.0, self._ap_global_step / max(warmup_steps, 1.0))
                ap_info["curriculum_mult"] = cur_mult

            # --- Module 1: Cone Cost (safety — always full weight, no curriculum) ---
            cone_cost_val = 0.0
            Z_matrix = None
            psi_per_agent = None
            if ap.get("enable_cone_cost", False) and self._y_cache is not None:
                cone_cost_val, cone_info, q_matrix = compute_group_cone_cost(
                    self.offensives, self.defensives, cfg, ap,
                    self._y_cache, self.current_step,
                    prev_q_matrix=self._ap_prev_q_matrix)
                self._ap_prev_q_matrix = q_matrix
                Z_matrix = cone_info.pop("_Z_matrix", None)
                psi_per_agent = cone_info.get("psi_agg_per_agent", [])
                w_cone = ap.get("cone_cost_weight", 1.0)  # NO curriculum on safety cost

                # Enhancement: per-agent cone cost (each agent gets own risk)
                if ap.get("per_agent_cone_cost", True) and psi_per_agent:
                    for i, off in enumerate(self.offensives):
                        if off.alive and i < len(psi_per_agent):
                            costs_list[i] += psi_per_agent[i] * w_cone
                else:
                    n_alive = max(sum(1 for o in self.offensives if o.alive), 1)
                    per_agent_cone = cone_cost_val * w_cone / n_alive
                    for i, off in enumerate(self.offensives):
                        if off.alive:
                            costs_list[i] += per_agent_cone

                # Cache for obs: Z_tilde_i, psi_agg_i
                Z_tilde_arr = cone_info.get("_Z_tilde", np.zeros(self.n_offensive))
                for i in range(self.n_offensive):
                    self._ap_obs_cache[i, 0] = np.clip(Z_tilde_arr[i] if i < len(Z_tilde_arr) else 0.0, -5, 5) / 5.0
                    self._ap_obs_cache[i, 1] = np.clip(psi_per_agent[i] if psi_per_agent and i < len(psi_per_agent) else 0.0, 0, 5) / 5.0
                ap_info.update(cone_info)

            # --- Module 2: Assignment Mismatch Reward ---
            mismatch_reward_val = 0.0
            if ap.get("enable_assignment_mismatch_reward", False):
                mismatch_reward_val, M_tilde, mismatch_info = compute_assignment_mismatch(
                    self.offensives, self.defensives, cfg, ap,
                    self._ap_fixed_assignment,
                    Z_matrix=Z_matrix,
                    prev_M_tilde=self._ap_prev_M_tilde)
                self._ap_prev_M_tilde = M_tilde
                self._ap_episode_max_M_tilde = max(self._ap_episode_max_M_tilde, M_tilde)
                w_mis = ap.get("mismatch_reward_weight", 0.3) * cur_mult
                # actual delta already scaled by lambda_M inside, rescale by curriculum
                scaled_mismatch = mismatch_reward_val * (cur_mult / max(ap.get("mismatch_reward_weight", 0.3), 1e-8)) * w_mis if ap.get("mismatch_reward_weight", 0.3) > 0 else 0.0

                # Enhancement: cooperative mismatch — credit individual contribution
                per_agent_contributions = mismatch_info.get("_per_agent_contribution", None)
                if ap.get("cooperative_mismatch", True) and per_agent_contributions is not None and scaled_mismatch > 0:
                    ind_w = ap.get("coop_mismatch_individual_weight", 0.6)
                    team_w = 1.0 - ind_w
                    n_alive = max(sum(1 for o in self.offensives if o.alive), 1)
                    team_share = scaled_mismatch * team_w / n_alive
                    total_contrib = sum(abs(c) for c in per_agent_contributions) + 1e-8
                    for i, off in enumerate(self.offensives):
                        if off.alive and i < len(per_agent_contributions):
                            ind_share = scaled_mismatch * ind_w * abs(per_agent_contributions[i]) / total_contrib
                            rewards_list[i] += ind_share + team_share
                else:
                    n_alive = max(sum(1 for o in self.offensives if o.alive), 1)
                    per_agent_mis = scaled_mismatch / n_alive
                    for i, off in enumerate(self.offensives):
                        if off.alive:
                            rewards_list[i] += per_agent_mis

                # Cache for obs: M_tilde normalized
                M_norm = np.clip(M_tilde / 50.0, 0, 1.0)  # normalize by empirical max
                for i in range(self.n_offensive):
                    self._ap_obs_cache[i, 2] = M_norm
                ap_info.update(mismatch_info)

            # --- Module 3: LOS Escape Reward ---
            escape_reward_val = 0.0
            if ap.get("enable_escape_reward", False):
                escape_reward_val, per_agent_esc, escape_info = compute_escape_reward(
                    self.offensives, self.defensives, cfg, ap)
                w_esc = cur_mult  # already weighted inside by lambda_E
                for i in range(len(self.offensives)):
                    if self.offensives[i].alive:
                        rewards_list[i] += per_agent_esc[i] * w_esc
                # Cache for obs: Xi_max per agent
                per_agent_xi = escape_info.get("_per_agent_Xi_max", None)
                if per_agent_xi is not None:
                    for i in range(min(len(per_agent_xi), self.n_offensive)):
                        self._ap_obs_cache[i, 3] = np.clip(per_agent_xi[i], 0, 2) / 2.0
                ap_info.update(escape_info)

            # --- Module 4: HVT Guidance + Soft Penetration Success Score ---
            if ap.get("enable_hvt_guidance", False):
                omega_ref = max(ap.get("hvt_omega_ref", 0.6), 1e-6)
                omega_dot_ref = max(ap.get("hvt_omega_dot_ref", 0.8), 1e-6)
                pn_hint_ref = max(ap.get("pn_hint_ref", 30.0), 1e-6)
                pn_nav_gain = ap.get("pn_nav_gain", 3.0)
                score_bias = ap.get("penetration_score_bias", -0.35)
                score_scale = ap.get("penetration_score_scale", 2.2)

                rho_values = []
                closing_values = []
                omega_values = []
                omega_dot_values = []
                pn_values = []
                score_values = []

                for i, off in enumerate(self.offensives):
                    if not off.alive or off.hit_hvt:
                        self._ap_penetration_score[i] = 0.0
                        if self._ap_guidance_obs_dim > 0 and self._ap_obs_dim >= 10:
                            self._ap_obs_cache[i, 4:10] = 0.0
                        continue

                    feats = compute_hvt_guidance_features(
                        off, self.hvt, self.dt,
                        prev_omega_los=self._ap_prev_hvt_omega[i],
                        pn_nav_gain=pn_nav_gain,
                    )
                    self._ap_prev_hvt_omega[i] = feats["omega_los"]

                    rho_norm = np.clip(feats["rho"] / max(cfg.get("obs_range", 5000.0), 1.0), 0.0, 1.0)
                    closing_norm = np.clip(feats["closing_speed"] / max(cfg.get("vel_range", 120.0), 1.0), -1.0, 1.0)
                    omega_norm = np.clip(feats["omega_los"] / omega_ref, 0.0, 1.0)
                    omega_dot_norm = np.clip(feats["omega_los_dot"] / omega_dot_ref, -1.0, 1.0)
                    pn_hint_norm = np.clip(feats["pn_hint"] / pn_hint_ref, -1.0, 1.0)

                    cone_risk_norm = self._ap_obs_cache[i, 1] if self._ap_obs_dim >= 2 else 0.0
                    mismatch_norm = self._ap_obs_cache[i, 2] if self._ap_obs_dim >= 3 else 0.0
                    detected_norm = np.clip(
                        off.detected_by_count / max(self.n_defensive, 1), 0.0, 1.0
                    )

                    pen_score = compute_penetration_success_score(
                        rho_norm=rho_norm,
                        closing_norm=closing_norm,
                        omega_norm=omega_norm,
                        omega_dot_norm=omega_dot_norm,
                        pn_hint_norm=pn_hint_norm,
                        cone_risk_norm=cone_risk_norm,
                        mismatch_norm=mismatch_norm,
                        detected_norm=detected_norm,
                        score_bias=score_bias,
                        score_scale=score_scale,
                    )
                    self._ap_penetration_score[i] = pen_score

                    if self._ap_guidance_obs_dim > 0 and self._ap_obs_dim >= 10:
                        self._ap_obs_cache[i, 4] = rho_norm
                        self._ap_obs_cache[i, 5] = closing_norm
                        self._ap_obs_cache[i, 6] = omega_norm
                        self._ap_obs_cache[i, 7] = omega_dot_norm
                        self._ap_obs_cache[i, 8] = pn_hint_norm
                        self._ap_obs_cache[i, 9] = pen_score

                    rho_values.append(feats["rho"])
                    closing_values.append(feats["closing_speed"])
                    omega_values.append(feats["omega_los"])
                    omega_dot_values.append(feats["omega_los_dot"])
                    pn_values.append(feats["pn_hint"])
                    score_values.append(pen_score)

                self._ap_team_penetration_score = float(np.mean(score_values)) if score_values else 0.0
                ap_info.update({
                    "penetration_success_score_mean": self._ap_team_penetration_score,
                    "penetration_success_score_max": float(np.max(score_values)) if score_values else 0.0,
                    "hvt_rho_mean": float(np.mean(rho_values)) if rho_values else 0.0,
                    "hvt_closing_speed_mean": float(np.mean(closing_values)) if closing_values else 0.0,
                    "hvt_omega_los_mean": float(np.mean(omega_values)) if omega_values else 0.0,
                    "hvt_omega_los_dot_mean": float(np.mean(omega_dot_values)) if omega_dot_values else 0.0,
                    "hvt_pn_hint_mean": float(np.mean(pn_values)) if pn_values else 0.0,
                })

                # --- Module 5: Attack Gate Reward (softly enabled by penetration score) ---
                self._ap_attack_gate_reward[:] = 0.0
                if ap.get("enable_attack_gate_reward", False):
                    gate_weight = ap.get("attack_gate_weight", 3.0)
                    w_prog = ap.get("attack_progress_weight", 1.0)
                    w_closing = ap.get("attack_closing_weight", 0.8)
                    w_los = ap.get("attack_los_weight", 0.5)
                    w_losdot = ap.get("attack_losdot_weight", 0.4)
                    w_pn = ap.get("attack_pn_align_weight", 0.6)
                    pn_accel_ref_g = max(ap.get("pn_accel_ref_g", 8.0), 1e-6)

                    gate_vals = []
                    r_prog_vals = []
                    r_closing_vals = []
                    r_los_vals = []
                    r_losdot_vals = []
                    r_pn_vals = []

                    for i, off in enumerate(self.offensives):
                        if not off.alive or off.hit_hvt:
                            continue

                        rho_norm = self._ap_obs_cache[i, 4] if self._ap_obs_dim >= 10 else 1.0
                        closing_norm = self._ap_obs_cache[i, 5] if self._ap_obs_dim >= 10 else 0.0
                        omega_norm = self._ap_obs_cache[i, 6] if self._ap_obs_dim >= 10 else 0.0
                        omega_dot_norm = self._ap_obs_cache[i, 7] if self._ap_obs_dim >= 10 else 0.0
                        pn_hint_norm = self._ap_obs_cache[i, 8] if self._ap_obs_dim >= 10 else 0.0

                        gate = float(np.clip(self._ap_penetration_score[i], 0.0, 1.0))

                        progress_term = 1.0 - np.clip(rho_norm, 0.0, 1.0)
                        closing_term = max(np.clip(closing_norm, -1.0, 1.0), 0.0)
                        los_term = 1.0 - np.clip(omega_norm, 0.0, 1.0)
                        losdot_term = 1.0 - min(abs(np.clip(omega_dot_norm, -1.0, 1.0)), 1.0)

                        a_cmd_g = np.sqrt(off.ny ** 2 + (off.nz - np.cos(off.gamma)) ** 2)
                        a_cmd_norm = np.clip(a_cmd_g / pn_accel_ref_g, 0.0, 1.0)
                        pn_target = max(np.clip(pn_hint_norm, -1.0, 1.0), 0.0)
                        pn_align_term = 1.0 - min(abs(a_cmd_norm - pn_target), 1.0)

                        attack_reward = gate_weight * cur_mult * gate * (
                            w_prog * progress_term
                            + w_closing * closing_term
                            + w_los * los_term
                            + w_losdot * losdot_term
                            + w_pn * pn_align_term
                        )
                        rewards_list[i] += attack_reward
                        self._ap_attack_gate_reward[i] = attack_reward

                        gate_vals.append(gate)
                        r_prog_vals.append(progress_term)
                        r_closing_vals.append(closing_term)
                        r_los_vals.append(los_term)
                        r_losdot_vals.append(losdot_term)
                        r_pn_vals.append(pn_align_term)

                    ap_info.update({
                        "attack_gate_mean": float(np.mean(gate_vals)) if gate_vals else 0.0,
                        "attack_reward_mean": float(np.mean(self._ap_attack_gate_reward)) if len(self._ap_attack_gate_reward) > 0 else 0.0,
                        "attack_reward_sum": float(np.sum(self._ap_attack_gate_reward)),
                        "attack_progress_term_mean": float(np.mean(r_prog_vals)) if r_prog_vals else 0.0,
                        "attack_closing_term_mean": float(np.mean(r_closing_vals)) if r_closing_vals else 0.0,
                        "attack_los_term_mean": float(np.mean(r_los_vals)) if r_los_vals else 0.0,
                        "attack_losdot_term_mean": float(np.mean(r_losdot_vals)) if r_losdot_vals else 0.0,
                        "attack_pn_align_term_mean": float(np.mean(r_pn_vals)) if r_pn_vals else 0.0,
                    })

        # 11. 终止
        done, done_reason = self._check_done()
        if done and done_reason == "timeout":
            for i in range(self.n_agents):
                rewards_list[i] += cfg["reward"]["timeout_penalty"]
            obs_range = max(cfg.get("obs_range", 5000.0), 1.0)
            timeout_dist_coef = cfg["reward"].get("timeout_distance_penalty_coef", 0.0)
            timeout_alive_pen = cfg["reward"].get("timeout_alive_penalty", 0.0)
            if timeout_dist_coef != 0.0 or timeout_alive_pen != 0.0:
                for i, off in enumerate(self.offensives):
                    if not off.alive or off.hit_hvt:
                        continue
                    dist_hvt = off.distance_to(self.hvt.x, self.hvt.y, self.hvt.z)
                    dist_ratio = np.clip(dist_hvt / obs_range, 0.0, 2.0)
                    rewards_list[i] -= timeout_dist_coef * dist_ratio
                    rewards_list[i] += timeout_alive_pen

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
        n_escaped = sum(1 for o in self.offensives if hasattr(o, '_n_escapes') and o._n_escapes > 0)

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
            # V11_01 新增
            "n_escapes_total": len(self.escape_events_total),
            "n_escaped_agents": n_escaped,
            "step_escapes": len(step_escapes),
            "step_misses": len(step_misses),
            "penetration_success_score_per_agent": self._ap_penetration_score.tolist(),
            "penetration_success_score_team": float(self._ap_team_penetration_score),
            "attack_gate_reward_per_agent": self._ap_attack_gate_reward.tolist(),
        }
        # Analytic priors info
        if self._ap_enabled and ap_info:
            info.update(ap_info)
            if done:
                info["ap_episode_max_M_tilde"] = self._ap_episode_max_M_tilde
                info["ap_cone_escape_success"] = self._ap_cone_escape_success
        if done:
            info["bad_transition"] = (done_reason == "timeout")

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
        V11_01 核心: 击杀/逃逸/命中判定

        击杀条件 (三级):
          Level 1: 纯碰撞 (dist < collision_kill_range=8m) → 无条件双杀
          Level 2: FOV跟踪击杀 (dist < kill_range=50m AND 连续FOV锁定N步) → 双杀
          Level 3: miss (在杀伤区内但目标不在FOV → 目标逃脱)

        逃逸条件:
          - 拦截器进入engagement_range后, 目标突破了FOV → 逃逸事件
          - 拦截器PN要求过载超出极限 → 自然产生FOV丢失 → 逃逸
          - 拦截器前向追踪放弃(目标飞到身后)  → 逃逸

        命中HVT:
          - 进攻方到HVT距离 < hit_hvt_range → 突防成功
        """
        cfg = self.config
        fov_half = cfg["fov_half_angle"]
        det_range = cfg["detection_range"]
        kill_range = cfg.get("kill_range", 50.0)
        collision_kill_range = cfg.get("collision_kill_range", 8.0)
        fov_escape_cfg = cfg.get("fov_escape", {})
        engagement_range = fov_escape_cfg.get("engagement_range", 200.0)
        min_tracking_steps = fov_escape_cfg.get("min_tracking_steps", 3)
        miss_cooldown_steps = fov_escape_cfg.get("miss_cooldown_steps", 50)
        fov_escape_enabled = fov_escape_cfg.get("enabled", True)

        step_hits = []
        step_escapes = []
        step_misses = []

        # ——————————————————————————————————
        # A. 拦截器 vs 进攻方: 击杀/逃逸判定
        # ——————————————————————————————————
        for di, d in enumerate(self.defensives):
            if not d.alive:
                continue
            policy = self.defensive_policies[di]

            for oi, off in enumerate(self.offensives):
                if not off.alive or off.hit_hvt:
                    continue

                pair_key = (di, oi)
                in_miss_cooldown = (pair_key in self.miss_cooldowns and
                                    self.miss_cooldowns[pair_key] > 0)

                dist = d.distance_3d(off)

                # --- Level 1: 纯碰撞击杀 ---
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

                # --- Level 2 & 3: FOV跟踪击杀 / FOV逃逸 ---
                if fov_escape_enabled and dist < kill_range:
                    target_in_fov = d.is_in_fov(off.x, off.y, off.z,
                                                fov_half, det_range)
                    # 更新交战跟踪
                    if pair_key not in self.engagement_tracking:
                        self.engagement_tracking[pair_key] = 0

                    if target_in_fov:
                        self.engagement_tracking[pair_key] += 1
                    else:
                        # 目标不在FOV! → MISS/逃逸!
                        prev_tracking = self.engagement_tracking.get(pair_key, 0)
                        if (not in_miss_cooldown) and (prev_tracking > 0 or dist < kill_range * 0.8):
                            # 之前有过锁定现在丢失, 或者很近但不在FOV = 逃逸
                            escape_ev = {
                                "off_idx": oi,
                                "def_idx": di,
                                "type": "fov_break",
                                "step": self.current_step,
                                "dist": dist,
                                "prev_tracking": prev_tracking,
                            }
                            step_escapes.append(escape_ev)
                            self.escape_events_total.append(escape_ev)
                            off._escaped_interceptor = True
                            off._n_escapes += 1

                            # 拦截器标记miss
                            policy.mark_target_missed(oi)
                            self.miss_cooldowns[pair_key] = miss_cooldown_steps
                            step_misses.append({
                                "off_idx": oi,
                                "def_idx": di,
                                "reason": "fov_break",
                            })

                        self.engagement_tracking[pair_key] = 0

                elif fov_escape_enabled and dist < engagement_range:
                    # 在交战区但未进入杀伤区 — 跟踪是否能保持FOV
                    target_in_fov = d.is_in_fov(off.x, off.y, off.z,
                                                fov_half, det_range)
                    if pair_key not in self.engagement_tracking:
                        self.engagement_tracking[pair_key] = 0

                    if target_in_fov:
                        self.engagement_tracking[pair_key] += 1
                    else:
                        # 在交战区丢失FOV — 检查拦截器是否过载饱和
                        sat_ny, sat_nz, sat_ratio = policy.is_overload_saturated()
                        if (not in_miss_cooldown) and (sat_ny or sat_nz):
                            # ★ 过载饱和导致的FOV丢失 = 逃逸! ★
                            escape_ev = {
                                "off_idx": oi,
                                "def_idx": di,
                                "type": "overload_escape",
                                "step": self.current_step,
                                "dist": dist,
                                "saturation_ratio": sat_ratio,
                                "demanded_ny": policy.demanded_ny,
                                "demanded_nz": policy.demanded_nz,
                            }
                            step_escapes.append(escape_ev)
                            self.escape_events_total.append(escape_ev)
                            off._escaped_interceptor = True
                            off._n_escapes += 1

                            policy.mark_target_missed(oi)
                            self.miss_cooldowns[pair_key] = miss_cooldown_steps
                            step_misses.append({
                                "off_idx": oi,
                                "def_idx": di,
                                "reason": "overload_saturation",
                                "sat_ratio": sat_ratio,
                            })

                        self.engagement_tracking[pair_key] = 0

                # 非FOV逃逸模式下，不进行非碰撞击杀（仅保留Level 1碰撞击杀）

            # 检查拦截器是否因前向限制放弃了目标 → 也算逃逸
            if (d.alive and
                policy.engagement_state in (InterceptorPolicy.STATE_ABANDONED,
                                            InterceptorPolicy.STATE_MISSED)):
                off_idx = policy.assigned_target_idx
                if off_idx is not None and 0 <= off_idx < self.n_offensive:
                    off = self.offensives[off_idx]
                    if off.alive and not off.hit_hvt:
                        # 检查是否已记录过该逃逸
                        pair_key = (di, off_idx)
                        if pair_key not in self.miss_cooldowns:
                            if policy.engagement_state == InterceptorPolicy.STATE_ABANDONED:
                                escape_ev = {
                                    "off_idx": off_idx,
                                    "def_idx": di,
                                    "type": "pass_through",
                                    "step": self.current_step,
                                }
                                step_escapes.append(escape_ev)
                                self.escape_events_total.append(escape_ev)
                                off._escaped_interceptor = True
                                off._n_escapes += 1
                            self.miss_cooldowns[pair_key] = miss_cooldown_steps
                            step_misses.append({
                                "off_idx": off_idx,
                                "def_idx": di,
                                "reason": "forward_pass",
                            })

        # ——————————————————————————————————
        # B. 命中HVT判定
        # ——————————————————————————————————
        hvt_range = cfg.get("hit_hvt_range", 500.0)
        for i, off in enumerate(self.offensives):
            if not off.alive or off.hit_hvt:
                continue
            if off.distance_to(self.hvt.x, self.hvt.y, self.hvt.z) < hvt_range:
                off.mark_hit_hvt()
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
        V11_01 观测空间 — 增加交战态势
        新增维度:
          - 防御方: overload饱和度, 交战状态(0/1 engaged)
          - 队友: escaped状态
          - 协同: 已逃脱拦截器数
        """
        cfg = self.config
        obs_range = cfg["obs_range"]
        vel_range = cfg["vel_range"]
        z_range = cfg["z_range"]

        hvt_x, hvt_y, hvt_z = self.hvt.x, self.hvt.y, self.hvt.z
        dists_to_hvt = []
        front_scores = []
        for off in self.offensives:
            if off.alive and not off.hit_hvt:
                d = off.distance_to(hvt_x, hvt_y, hvt_z)
                dists_to_hvt.append(d)
                front_scores.append(d)
            else:
                dists_to_hvt.append(float('inf'))
                front_scores.append(float('inf'))

        sorted_by_front = sorted(range(self.n_offensive), key=lambda i: front_scores[i])
        front_rank = [0] * self.n_offensive
        for rank, idx in enumerate(sorted_by_front):
            front_rank[idx] = rank

        alive_dists = [d for d in dists_to_hvt if d < float('inf')]
        team_min_dist = min(alive_dists) if alive_dists else float('inf')
        def_alive_ratio = sum(1 for d in self.defensives if d.alive) / max(self.n_defensive, 1)

        # V11_01: 已逃脱拦截器的进攻方数量
        n_escaped_agents = sum(1 for o in self.offensives
                              if hasattr(o, '_n_escapes') and o._n_escapes > 0 and o.alive)

        obs_list = []
        for ai, agent in enumerate(self.offensives):
            obs = []
            # === 自身状态 (9) ===
            obs.extend([
                agent.x / obs_range, agent.y / obs_range, agent.z / z_range,
                agent.v / vel_range, agent.heading / np.pi,
                agent.gamma / (np.pi / 4),
                agent.nx / 4.0, agent.ny / 5.0, agent.nz / 3.0,
            ])
            # === HVT相对位置 (3) ===
            obs.extend([
                (hvt_x - agent.x) / obs_range,
                (hvt_y - agent.y) / obs_range,
                (hvt_z - agent.z) / z_range,
            ])
            # === 防御方信息 (11*K) ===
            if agent.alive:
                def_dists = []
                for di, d in enumerate(self.defensives):
                    dd = d.distance_3d(agent) if d.alive else float('inf')
                    def_dists.append((dd, di))
                def_dists.sort()
                for k in range(self.obs_k_def):
                    if k < len(def_dists) and def_dists[k][0] < float('inf'):
                        d = self.defensives[def_dists[k][1]]
                        di_idx = def_dists[k][1]
                        obs.extend([
                            (d.x - agent.x) / obs_range,
                            (d.y - agent.y) / obs_range,
                            (d.z - agent.z) / z_range,
                            (d.v - agent.v) / vel_range,
                            (d.heading - agent.heading) / np.pi,
                            (d.gamma - agent.gamma) / (np.pi / 4),
                            1.0,
                        ])
                        # threat_heading
                        dx_me = agent.x - d.x
                        dy_me = agent.y - d.y
                        angle_to_me = np.arctan2(dy_me, dx_me)
                        heading_diff = angle_to_me - d.heading
                        heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
                        threat_heading = np.cos(heading_diff)
                        obs.append(threat_heading)

                        # closing_speed
                        dist_d = max(def_dists[k][0], 1.0)
                        cos_gd = np.cos(d.gamma)
                        vx_d = d.v * cos_gd * np.cos(d.heading)
                        vy_d = d.v * cos_gd * np.sin(d.heading)
                        vz_d = d.v * np.sin(d.gamma)
                        cos_ga = np.cos(agent.gamma)
                        vx_a = agent.v * cos_ga * np.cos(agent.heading)
                        vy_a = agent.v * cos_ga * np.sin(agent.heading)
                        vz_a = agent.v * np.sin(agent.gamma)
                        rdx = agent.x - d.x
                        rdy = agent.y - d.y
                        rdz = agent.z - d.z
                        closing_v = -((rdx*(vx_a-vx_d) + rdy*(vy_a-vy_d) + rdz*(vz_a-vz_d)) / dist_d)
                        obs.append(np.clip(closing_v / vel_range, -1.0, 1.0))

                        # V11_01 新增: 拦截器过载饱和度
                        policy = self.defensive_policies[di_idx]
                        _, _, sat_ratio = policy.is_overload_saturated()
                        obs.append(np.clip(sat_ratio, 0.0, 3.0) / 3.0)

                        # V11_01 新增: 是否在交战中 (0或1)
                        is_engaged = 1.0 if (policy.engagement_state ==
                                            InterceptorPolicy.STATE_ENGAGED) else 0.0
                        obs.append(is_engaged)
                    else:
                        obs.extend([0.0] * 11)
            else:
                obs.extend([0.0] * (11 * self.obs_k_def))

            # === 队友信息 (10*(n_off-1)) ===
            for aj, teammate in enumerate(self.offensives):
                if aj == ai:
                    continue
                if teammate.alive and agent.alive:
                    obs.extend([
                        (teammate.x - agent.x) / obs_range,
                        (teammate.y - agent.y) / obs_range,
                        (teammate.z - agent.z) / z_range,
                        teammate.v / vel_range,
                        teammate.heading / np.pi,
                        teammate.gamma / (np.pi / 4),
                        1.0,
                    ])
                    mate_dist = dists_to_hvt[aj] / obs_range if dists_to_hvt[aj] < float('inf') else 1.0
                    obs.append(mate_dist)
                    obs.append(front_rank[aj] / max(self.n_offensive - 1, 1))
                    # V11_01 新增: 队友是否已逃脱过拦截器
                    obs.append(1.0 if (hasattr(teammate, '_escaped_interceptor')
                                       and teammate._escaped_interceptor) else 0.0)
                else:
                    obs.extend([0.0] * 10)

            # === 暴露状态 (3) ===
            obs.append(1.0 if agent.detected else 0.0)
            obs.append(agent.detected_by_count / max(self.n_defensive, 1))
            obs.append(min(agent.continuous_exposure / 50.0, 1.0))

            # === 全局信息 (2) ===
            dist_hvt_norm = dists_to_hvt[ai] / obs_range if dists_to_hvt[ai] < float('inf') else 1.0
            obs.append(dist_hvt_norm)
            obs.append(self.current_step / self.max_steps)

            # === 协同态势 (5) ===
            obs.append(front_rank[ai] / max(self.n_offensive - 1, 1))

            n_threats = 0
            if agent.alive:
                for d in self.defensives:
                    if not d.alive:
                        continue
                    dx_me = agent.x - d.x
                    dy_me = agent.y - d.y
                    angle_to_me = np.arctan2(dy_me, dx_me)
                    hdiff = angle_to_me - d.heading
                    hdiff = np.arctan2(np.sin(hdiff), np.cos(hdiff))
                    if abs(hdiff) < np.deg2rad(30.0) and d.distance_3d(agent) < 3000.0:
                        n_threats += 1
            obs.append(n_threats / max(self.n_defensive, 1))

            obs.append(team_min_dist / obs_range if team_min_dist < float('inf') else 1.0)
            obs.append(def_alive_ratio)

            # V11_01 新增: 全队已逃脱拦截器的数量(归一化)
            obs.append(n_escaped_agents / max(self.n_offensive, 1))

            # === 解析先验观测 (base4 + optional6 guidance dims) ===
            if self._ap_obs_dim > 0:
                obs.extend(self._ap_obs_cache[ai].tolist())

            obs_list.append(np.array(obs, dtype=np.float32))
        return obs_list

    def _get_share_obs(self):
        cfg = self.config
        obs_range = cfg["obs_range"]
        vel_range = cfg["vel_range"]
        z_range = cfg["z_range"]
        so = []
        for off in self.offensives:
            so.extend([
                off.x / obs_range, off.y / obs_range, off.z / z_range,
                off.v / vel_range, off.heading / np.pi,
                off.gamma / (np.pi / 4),
                off.nx / 4.0, off.ny / 5.0, off.nz / 3.0,
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
        if self._ap_share_extra_dim > 0:
            so.append(float(self._ap_team_penetration_score))
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
