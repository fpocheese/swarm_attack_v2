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
        # V22: Decoy game state
        self._ap_prev_Phi_decoy = None
        # V22: Effective penetration state
        self._ap_prev_N_eff = None

    def _compute_space_dims(self):
        cfg = self.config
        n_def = cfg["n_defensive"]
        n_off = cfg["n_offensive"]
        self.obs_k_def = min(n_def, 4)
        # V22 观测空间 — 锁定态势信息
        # self: 9, hvt: 3
        # 防御方(最近K个): 每个11维 (包含: is_chasing_me, lock_state)
        # 队友: 每个10维 (包含: 被追击拦截器数)
        # 暴露: 3, 全局: 2
        # 协同态势: 5 (包含: 被追击拦截器数(自身))
        # V23: HVT相对位置3 + LOS视线角速度2 = 5 (was 3)
        self._base_obs_dim = 9 + 5 + 11 * self.obs_k_def + 10 * (n_off - 1) + 3 + 2 + 5
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
        self.prev_team_min_dist = min(self.prev_dists_to_hvt)
        self.hit_count = 0
        self.hit_indices = []
        self.kill_events = []
        self.escape_events_total = []
        self.engagement_tracking = {}
        self.miss_cooldowns = {}
        self.lock_events_log = []

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
            # V22: Decoy game + effective penetration per-episode state
            self._ap_prev_Phi_decoy = None
            self._ap_prev_N_eff = None

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
                ax_cmd, ay_cmd, mu_cmd = policy.get_action(self.offensives, self.dt)
                d.step(ax_cmd, ay_cmd, mu_cmd, self.dt)

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
        step_lock_events = []
        for di, policy in enumerate(self.defensive_policies):
            if self.defensives[di].alive:
                lock_ev = policy.update_lock_state(
                    self.offensives, self.current_step)
                if lock_ev is not None:
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
            miss_events=step_misses,
            defensive_policies=self.defensive_policies)

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

            # --- Module 2b: Decoy Game (V22 替代旧 Assignment Mismatch) ---
            if ap.get("enable_decoy_game", False):
                decoy_reward, per_agent_decoy, Phi_decoy, decoy_info = compute_decoy_game(
                    self.offensives, self.defensives, self.hvt, cfg, ap,
                    locked_by_map=self.locked_by_map,
                    prev_Phi_decoy=self._ap_prev_Phi_decoy)
                self._ap_prev_Phi_decoy = Phi_decoy

                for i in range(self.n_offensive):
                    if self.offensives[i].alive:
                        rewards_list[i] += per_agent_decoy[i] * cur_mult
                ap_info.update(decoy_info)

            # --- Module 3b: Effective Penetration (V22) ---
            if ap.get("enable_effective_penetration", False):
                # 收集各模块的 per-agent 风险/能力值
                _cone_risk = [self._ap_obs_cache[i, 1] * 5.0
                              for i in range(self.n_offensive)]  # psi_agg_i (un-normalized)
                _lock_pressure = (decoy_info.get("lock_pressure_per_agent", [0.0] * self.n_offensive)
                                  if ap.get("enable_decoy_game", False)
                                  else [0.0] * self.n_offensive)
                _locked_by_count = [len(self.locked_by_map.get(i, []))
                                    for i in range(self.n_offensive)]
                _E_i_esc = (escape_info.get("E_i_esc", [0.0] * self.n_offensive)
                            if ap.get("enable_escape_reward", False)
                            else [0.0] * self.n_offensive)

                pen_reward, per_agent_pen, N_eff, pen_info = compute_effective_penetration(
                    self.offensives, self.defensives, self.hvt, cfg, ap,
                    cone_risk_per_agent=_cone_risk,
                    lock_pressure_per_agent=_lock_pressure,
                    locked_by_count_per_agent=_locked_by_count,
                    E_i_esc_per_agent=_E_i_esc,
                    prev_N_eff=self._ap_prev_N_eff)
                self._ap_prev_N_eff = N_eff

                for i in range(self.n_offensive):
                    if self.offensives[i].alive:
                        rewards_list[i] += per_agent_pen[i] * cur_mult
                ap_info.update(pen_info)

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

                        a_cmd_g = off.ay / G
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
        hvt_range = cfg.get("hit_hvt_range", 50.0)
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
        V22 观测空间 — 增加锁定态势
        新增维度:
          - 防御方: overload饱和度, 锁定状态(LOCKED与否)
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

        # V24: 统计每个进攻方被多少拦截器追击（用于诱饵观测）
        # (n_escaped_agents 已不再使用，保留变量名兼容info dict)

        obs_list = []
        for ai, agent in enumerate(self.offensives):
            obs = []
            # === 自身状态 (9) ===
            obs.extend([
                agent.x / obs_range, agent.y / obs_range, agent.z / z_range,
                agent.v / vel_range, agent.heading / np.pi,
                agent.gamma / (np.pi / 4),
                agent.ax / 20.0, agent.ay / 25.0, agent.mu / np.pi,
            ])
            # === HVT相对位置 (3) ===
            rel_x = hvt_x - agent.x
            rel_y = hvt_y - agent.y
            rel_z = hvt_z - agent.z
            obs.extend([
                rel_x / obs_range,
                rel_y / obs_range,
                rel_z / z_range,
            ])
            # === HVT视线角速度 (2) — V23新增，供PN制导学习 ===
            # LOS rate: d(lambda_az)/dt, d(lambda_el)/dt
            # HVT静止，相对速度 = -自身速度
            # r_dot = v_HVT - v_self = -v_self (HVT静止)
            _los_rate_az = 0.0
            _los_rate_el = 0.0
            if agent.alive and hasattr(agent, '_prev_pos') and self.dt > 0:
                px, py, pz = agent._prev_pos
                # 上一步视线向量 (HVT - prev_pos)
                r0x, r0y, r0z = hvt_x - px, hvt_y - py, hvt_z - pz
                r0_xy = max(np.sqrt(r0x**2 + r0y**2), 1e-6)
                r0_mag = max(np.sqrt(r0x**2 + r0y**2 + r0z**2), 1e-6)
                # 当前视线向量
                r1x, r1y, r1z = rel_x, rel_y, rel_z
                r1_xy = max(np.sqrt(r1x**2 + r1y**2), 1e-6)
                r1_mag = max(np.sqrt(r1x**2 + r1y**2 + r1z**2), 1e-6)
                # 方位角变化率 (水平面内)
                lam0_az = np.arctan2(r0y, r0x)
                lam1_az = np.arctan2(r1y, r1x)
                d_lam_az = np.arctan2(np.sin(lam1_az - lam0_az), np.cos(lam1_az - lam0_az))
                _los_rate_az = np.clip(d_lam_az / (self.dt + 1e-9) / 0.5, -1.0, 1.0)  # 归一化 0.5rad/s
                # 仰角变化率
                lam0_el = np.arctan2(r0z, r0_xy)
                lam1_el = np.arctan2(r1z, r1_xy)
                d_lam_el = np.arctan2(np.sin(lam1_el - lam0_el), np.cos(lam1_el - lam0_el))
                _los_rate_el = np.clip(d_lam_el / (self.dt + 1e-9) / 0.5, -1.0, 1.0)
            obs.extend([_los_rate_az, _los_rate_el])
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

                        # V24: 这个拦截器是否在追击我 (诱饵关键信息)
                        policy = self.defensive_policies[di_idx]
                        is_chasing_me = 1.0 if (policy.target is agent) else 0.0
                        obs.append(is_chasing_me)

                        # V24: 拦截器是否处于LOCKED状态 (0或1)
                        is_locked = 1.0 if (policy.lock_mode ==
                                            InterceptorPolicy.STATE_LOCKED) else 0.0
                        obs.append(is_locked)
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
                    # V24: 队友是否正在吸引拦截器(被追击)
                    n_chasers = 0
                    for dp in self.defensive_policies:
                        if (dp.interceptor.alive and dp.target is teammate):
                            n_chasers += 1
                    obs.append(n_chasers / max(self.n_defensive, 1))
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

            # V24: 被追击的拦截器数量(归一化) — 替代旧的n_escaped_agents
            n_chasers_self = 0
            if agent.alive:
                for dp in self.defensive_policies:
                    if dp.interceptor.alive and dp.target is agent:
                        n_chasers_self += 1
            obs.append(n_chasers_self / max(self.n_defensive, 1))

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
                off.ax / 20.0, off.ay / 25.0, off.mu / np.pi,
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
