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
        # self: 9, hvt: 3, def: 7*K, teammates: 7*(n_off-1),
        # exposure: 3, global: 2,
        # 协同信息: n_locked_on_me(1) + my_hvt_rank(1) + teammate_roles(3*(n_off-1))
        #   teammate_roles per mate: [dist_hvt_rank/n_off, n_locked, is_closer_to_hvt]
        self.coop_dim = 2 + 3 * (n_off - 1)
        self.obs_dim = 9 + 3 + 7 * self.obs_k_def + 7 * (n_off - 1) + 3 + 2 + self.coop_dim
        # share_obs: all_off: 10*n_off, all_def: 7*n_def, hvt: 3, relations: 4,
        #   lock_map: n_off (each: how many defenders locked on), team_min_dist: 1
        self.share_obs_dim = 10 * n_off + 7 * n_def + 3 + 4 + n_off + 1

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
        cfg = self.config
        obs_range = cfg["obs_range"]
        vel_range = cfg["vel_range"]
        z_range = cfg["z_range"]

        # 预计算协同信息
        dists_to_hvt = []
        for off in self.offensives:
            if off.alive and not off.hit_hvt:
                dists_to_hvt.append(off.distance_to(self.hvt.x, self.hvt.y, self.hvt.z))
            else:
                dists_to_hvt.append(float('inf'))
        # 排名 (0=最近, n-1=最远)
        sorted_indices = sorted(range(self.n_offensive), key=lambda i: dists_to_hvt[i])
        hvt_rank = [0] * self.n_offensive
        for rank, idx in enumerate(sorted_indices):
            hvt_rank[idx] = rank

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
                (self.hvt.x - agent.x) / obs_range,
                (self.hvt.y - agent.y) / obs_range,
                (self.hvt.z - agent.z) / z_range,
            ])
            # === 防御方信息 (7*K) ===
            if agent.alive:
                def_dists = []
                for di, d in enumerate(self.defensives):
                    dd = d.distance_3d(agent) if d.alive else float('inf')
                    def_dists.append((dd, di))
                def_dists.sort()
                for k in range(self.obs_k_def):
                    if k < len(def_dists) and def_dists[k][0] < float('inf'):
                        d = self.defensives[def_dists[k][1]]
                        obs.extend([
                            (d.x - agent.x) / obs_range,
                            (d.y - agent.y) / obs_range,
                            (d.z - agent.z) / z_range,
                            (d.v - agent.v) / vel_range,
                            (d.heading - agent.heading) / np.pi,
                            (d.gamma - agent.gamma) / (np.pi / 4),
                            1.0,
                        ])
                    else:
                        obs.extend([0.0] * 7)
            else:
                obs.extend([0.0] * (7 * self.obs_k_def))
            # === 队友信息 (7*(n_off-1)) ===
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
                else:
                    obs.extend([0.0] * 7)
            # === 暴露状态 (3) ===
            obs.append(1.0 if agent.detected else 0.0)
            obs.append(agent.detected_by_count / max(self.n_defensive, 1))
            obs.append(min(agent.continuous_exposure / 50.0, 1.0))
            # === 全局信息 (2) ===
            dist_hvt = agent.distance_to(self.hvt.x, self.hvt.y, self.hvt.z) / obs_range if agent.alive else 1.0
            obs.append(dist_hvt)
            obs.append(self.current_step / self.max_steps)

            # === 协同突防信息 (coop_dim) ===
            # 1. 被几架拦截器锁定 (归一化到0~1)
            n_locked = len(self.lock_on_map.get(ai, []))
            obs.append(n_locked / max(self.n_defensive, 1))
            # 2. 我的HVT距离排名 (0=最近→值最小, 鼓励排名低的冲)
            obs.append(hvt_rank[ai] / max(self.n_offensive - 1, 1))
            # 3. 每个队友的协同信息 (3 per teammate)
            for aj, teammate in enumerate(self.offensives):
                if aj == ai:
                    continue
                if teammate.alive and agent.alive:
                    # 队友的HVT距离排名
                    mate_rank = hvt_rank[aj] / max(self.n_offensive - 1, 1)
                    # 队友被几架拦截器锁定
                    mate_locked = len(self.lock_on_map.get(aj, [])) / max(self.n_defensive, 1)
                    # 队友是否比我更近HVT (1=是, 可能是突防者)
                    mate_closer = 1.0 if dists_to_hvt[aj] < dists_to_hvt[ai] else 0.0
                    obs.extend([mate_rank, mate_locked, mate_closer])
                else:
                    obs.extend([1.0, 0.0, 0.0])

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
        so.extend([min_dist, n_det / max(self.n_offensive, 1), off_alive_ratio, def_alive_ratio])
        # 协同信息: 每架进攻机被锁定的拦截器数
        for i in range(self.n_offensive):
            so.append(len(self.lock_on_map.get(i, [])) / max(self.n_defensive, 1))
        # 全队最近HVT距离
        so.append(min_dist)
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
