"""
FOV Penetration Environment V3
================================
三维同构集群协同突防环境
- 进攻方同构: 任何一架都能命中 HVT
- 三维动力学: (x,y,z,v,heading,gamma), 动作 3 维 (nx,ny,nz)
- 防御方: 匈牙利目标分配 + 3D PN
- 击杀规则: 统一 3m
"""

import numpy as np
from gym.spaces import Box

from .config import get_config, G
from .entities import Aircraft, HVT
from .dynamics import action_to_overload_3d
from .reward_cost import compute_rewards, compute_costs
from .policies_interceptor import InterceptorPolicy
from .target_assignment import assign_targets


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
        self.assignments = {}
        # 协同突防信息
        self.lock_on_map = {}       # {off_idx: [def_idx, ...]} 谁在追谁
        self.prev_team_min_dist = None  # 上一步全队最近HVT距离

    def _compute_space_dims(self):
        cfg = self.config
        n_def = cfg["n_defensive"]
        n_off = cfg["n_offensive"]
        self.obs_k_def = min(n_def, 4)
        # === V10 观测空间 ===
        # self: 9, hvt: 3
        # 防御方(按距离排序最近K个): 每个9维
        #   [dx,dy,dz, dv, dheading, dgamma, alive,
        #    threat_heading(是否朝我飞), closing_speed(接近速度)]
        # 队友: 每个9维
        #   [dx,dy,dz, v, heading, gamma, alive,
        #    mate_dist_hvt(队友到HVT距离), mate_front_rank(队友前后排名)]
        # 暴露: 3, 全局: 2
        # 协同态势: 4
        #   [my_front_rank, n_threats_on_me, team_min_dist_hvt, def_alive_ratio]
        self.obs_dim = 9 + 3 + 9 * self.obs_k_def + 9 * (n_off - 1) + 3 + 2 + 4
        # share_obs保持简洁: all_off + all_def + hvt + global
        self.share_obs_dim = 10 * n_off + 7 * n_def + 3 + 5

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
        """
        计算锁定关系图: 哪些拦截器在追哪架进攻机
        基于当前分配 + 拦截器存活状态
        """
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

        # 4. 击杀与命中 — 记录本步新死亡
        alive_before_off = [off.alive for off in self.offensives]
        alive_before_def = [d.alive for d in self.defensives]
        step_hits = self._check_kills_and_hits()

        # 5. 边界与撞地
        self._check_boundary_and_ground()

        # 计算本步刚刚死亡的列表
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

        # 8. 奖励 — 传入协同信息
        rewards_list, reward_info = compute_rewards(
            self.offensives, self.defensives, self.hvt, cfg,
            self.prev_dists_to_hvt, hit_events=step_hits,
            current_step=self.current_step,
            just_killed=just_killed_off,
            just_killed_def=just_killed_def,
            lock_on_map=self.lock_on_map,
            prev_team_min_dist=self.prev_team_min_dist)

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

        # 9. 成本
        costs_list, cost_info = compute_costs(self.offensives, self.defensives, cfg)

        # 10. 终止
        done, done_reason = self._check_done()
        if done and done_reason == "timeout":
            for i in range(self.n_agents):
                rewards_list[i] += cfg["reward"]["timeout_penalty"]

        # 11. 组装
        rewards = [[r] for r in rewards_list]
        costs = [[c] for c in costs_list]
        dones = [done] * self.n_agents

        off_alive = sum(1 for o in self.offensives if o.alive)
        def_alive = sum(1 for d in self.defensives if d.alive)
        avg_exposure = np.mean([o.total_exposure_steps for o in self.offensives])
        first_det_steps = [o.first_detected_step for o in self.offensives if o.first_detected_step >= 0]
        first_det_avg = np.mean(first_det_steps) if first_det_steps else -1.0
        n_detected_now = sum(1 for o in self.offensives if o.alive and o.detected)

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
        }
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

    def _check_kills_and_hits(self):
        kill_range = self.config["kill_range"]
        step_hits = []
        # 碰撞击杀：拦截器与进攻方碰撞时，双方同归于尽
        for d in self.defensives:
            if not d.alive:
                continue
            for off in self.offensives:
                if not off.alive or off.hit_hvt:
                    continue
                if d.distance_3d(off) < kill_range:
                    off.kill()
                    d.kill()  # 拦截器碰撞后也丧失战斗力
                    self.kill_events.append({
                        "step": self.current_step,
                        "defensive_id": d.uid,
                        "offensive_id": off.uid,
                        "mutual_kill": True,
                    })
                    break  # 该防御方已死，不再检查其他进攻方
        # 命中HVT判定
        for i, off in enumerate(self.offensives):
            if not off.alive or off.hit_hvt:
                continue
            if off.distance_to(self.hvt.x, self.hvt.y, self.hvt.z) < kill_range:
                off.mark_hit_hvt()
                self.hit_count += 1
                self.hit_indices.append(i)
                step_hits.append(i)
        return step_hits

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
        V10 观测空间 — 态势推断代替信息泄露
        进攻方不知道敌方分配方案，但可以通过以下态势推断威胁:
        - 拦截器heading是否朝我飞 (threat_heading)
        - 拦截器相对接近速度 (closing_speed)
        - 队友在编队中的前后位置 (front_rank)
        """
        cfg = self.config
        obs_range = cfg["obs_range"]
        vel_range = cfg["vel_range"]
        z_range = cfg["z_range"]

        # 预计算: 每架到HVT的距离 + 沿HVT方向的前后排名
        hvt_x, hvt_y, hvt_z = self.hvt.x, self.hvt.y, self.hvt.z
        dists_to_hvt = []
        # "前方"定义: 沿进攻方向(朝HVT)的投影距离越小=越前方
        front_scores = []
        for off in self.offensives:
            if off.alive and not off.hit_hvt:
                d = off.distance_to(hvt_x, hvt_y, hvt_z)
                dists_to_hvt.append(d)
                front_scores.append(d)  # 距离越小=越前方
            else:
                dists_to_hvt.append(float('inf'))
                front_scores.append(float('inf'))

        # 前后排名 (0=最前/最近HVT, n-1=最后)
        sorted_by_front = sorted(range(self.n_offensive), key=lambda i: front_scores[i])
        front_rank = [0] * self.n_offensive
        for rank, idx in enumerate(sorted_by_front):
            front_rank[idx] = rank

        # 全队最近HVT距离
        alive_dists = [d for d in dists_to_hvt if d < float('inf')]
        team_min_dist = min(alive_dists) if alive_dists else float('inf')

        # 防守方存活比例
        def_alive_ratio = sum(1 for d in self.defensives if d.alive) / max(self.n_defensive, 1)

        obs_list = []
        for ai, agent in enumerate(self.offensives):
            obs = []
            # === 自身状态 (9) ===
            obs.extend([
                agent.x / obs_range, agent.y / obs_range, agent.z / z_range,
                agent.v / vel_range, agent.heading / np.pi,
                agent.gamma / (np.pi / 4),
                agent.nx / 4.0, agent.ny / 4.0, agent.nz / 3.0,
            ])
            # === HVT相对位置 (3) ===
            obs.extend([
                (hvt_x - agent.x) / obs_range,
                (hvt_y - agent.y) / obs_range,
                (hvt_z - agent.z) / z_range,
            ])
            # === 防御方信息 (9*K) — 含态势推断! ===
            if agent.alive:
                def_dists = []
                for di, d in enumerate(self.defensives):
                    dd = d.distance_3d(agent) if d.alive else float('inf')
                    def_dists.append((dd, di))
                def_dists.sort()
                for k in range(self.obs_k_def):
                    if k < len(def_dists) and def_dists[k][0] < float('inf'):
                        d = self.defensives[def_dists[k][1]]
                        # 基础相对状态
                        obs.extend([
                            (d.x - agent.x) / obs_range,
                            (d.y - agent.y) / obs_range,
                            (d.z - agent.z) / z_range,
                            (d.v - agent.v) / vel_range,
                            (d.heading - agent.heading) / np.pi,
                            (d.gamma - agent.gamma) / (np.pi / 4),
                            1.0,
                        ])
                        # === 态势推断: 该拦截器是否在追我? ===
                        # threat_heading: 拦截器heading指向我的程度 [-1,1]
                        dx_me = agent.x - d.x
                        dy_me = agent.y - d.y
                        angle_to_me = np.arctan2(dy_me, dx_me)
                        heading_diff = angle_to_me - d.heading
                        heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
                        threat_heading = np.cos(heading_diff)  # 1=正对我飞, -1=背对
                        obs.append(threat_heading)

                        # closing_speed: 相对接近速度 (正=在靠近)
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
                    else:
                        obs.extend([0.0] * 9)
            else:
                obs.extend([0.0] * (9 * self.obs_k_def))

            # === 队友信息 (9*(n_off-1)) — 含编队位置! ===
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
                    # 队友到HVT距离 (归一化)
                    mate_dist = dists_to_hvt[aj] / obs_range if dists_to_hvt[aj] < float('inf') else 1.0
                    obs.append(mate_dist)
                    # 队友前后排名 (0=最前, 归一化)
                    obs.append(front_rank[aj] / max(self.n_offensive - 1, 1))
                else:
                    obs.extend([0.0] * 9)

            # === 暴露状态 (3) ===
            obs.append(1.0 if agent.detected else 0.0)
            obs.append(agent.detected_by_count / max(self.n_defensive, 1))
            obs.append(min(agent.continuous_exposure / 50.0, 1.0))

            # === 全局信息 (2) ===
            dist_hvt_norm = dists_to_hvt[ai] / obs_range if dists_to_hvt[ai] < float('inf') else 1.0
            obs.append(dist_hvt_norm)
            obs.append(self.current_step / self.max_steps)

            # === 协同态势 (4) — 纯态势推断, 无信息泄露! ===
            # 1. 我的前后排名 (0=最前/最近HVT → 最可能是突防者)
            obs.append(front_rank[ai] / max(self.n_offensive - 1, 1))
            # 2. 有多少拦截器"看起来在追我" (通过态势推断)
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
            # 3. 全队最近HVT距离
            obs.append(team_min_dist / obs_range if team_min_dist < float('inf') else 1.0)
            # 4. 防守方存活比例
            obs.append(def_alive_ratio)

            obs_list.append(np.array(obs, dtype=np.float32))
        return obs_list

    def _get_share_obs(self):
        """V10 共享观测 — 不暴露敌方分配方案"""
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
                off.nx / 4.0, off.ny / 4.0, off.nz / 3.0,
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
