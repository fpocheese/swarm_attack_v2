"""
FOV Penetration Environment V2
================================
基于敌方视场角(FOV)限制的固定翼无人机集群协同突防

击杀规则 (统一3m):
  - 拦截器 → 进攻/护卫飞行器: 距离<3m → 进攻/护卫被击杀
  - 护卫飞行器 → 拦截器: 距离<3m → 拦截器被击杀
  - 进攻飞行器 → HVT: 距离<3m → 突防成功

胜负判定:
  - 进攻方获胜: 进攻飞行器成功打击HVT (距离<3m)
  - 防御方获胜: 进攻飞行器被拦截
  - 超时: 进攻方未能突防, 视为失败

训练目标: 进攻集群 (1 attacker + 3 escorts) 成功突防
"""

import numpy as np
from gym.spaces import Box

from .config import get_config, G
from .entities import Aircraft, HVT
from .dynamics import action_to_overload
from .reward_cost import compute_rewards, compute_costs
from .policies_interceptor import InterceptorPolicy


class FOVPenetrationEnv:
    """
    多智能体协同突防环境 V2
    """

    def __init__(self, config=None):
        self.config = get_config(config)
        cfg = self.config

        self.n_attackers = cfg["n_attackers"]
        self.n_escorts = cfg["n_escorts"]
        self.n_interceptors = cfg["n_interceptors"]
        self.n_agents = self.n_attackers + self.n_escorts

        self.hvt = HVT(cfg["hvt_position"][0], cfg["hvt_position"][1])
        self.attacker = None
        self.escorts = []
        self.interceptors = []
        self.interceptor_policies = []

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
            Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            for _ in range(self.n_agents)
        ])

        self.current_step = 0
        self.max_steps = cfg["max_steps"]
        self.dt = cfg["dt"]
        self._seed = None
        self.rng = np.random.RandomState()

        # 统计
        self.prev_dist_to_hvt = None
        self.exposure_steps_attacker = 0
        self.cumulative_cost = 0.0
        self.fov_violation_count = 0
        self.intercept_events = []
        self.escort_kill_events = []  # 护卫击杀拦截器事件

    def _compute_space_dims(self):
        cfg = self.config
        n_intc = cfg["n_interceptors"]
        # 局部观测:
        # 自身: x,y,v,heading,nx,ny = 6
        # HVT: dx,dy = 2
        # 拦截器: (dx,dy,dv,dheading,alive) * n_intc = 5*n_intc
        # FOV状态: self_in_fov, attacker_exposed = 2
        # 队友: (dx,dy,v,heading,alive) * (n_agents-1) = 5*(n_agents-1)
        # 最近威胁距离: 1
        # 到HVT距离归一化: 1
        self.obs_dim = 6 + 2 + 5 * n_intc + 2 + 5 * (self.n_agents - 1) + 1 + 1

        # 共享观测:
        # 所有训练agent: (x,y,v,heading,nx,ny,alive) * n_agents = 7*n_agents
        # 所有拦截器: (x,y,v,heading,alive) * n_intc = 5*n_intc
        # HVT: x,y = 2
        # 攻防关系: dist_atk_hvt, atk_fov_count, escorts_alive, intc_alive = 4
        self.share_obs_dim = 7 * self.n_agents + 5 * n_intc + 2 + 4

    def seed(self, seed=None):
        self._seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed)

    def _create_entities(self):
        cfg = self.config

        ai = cfg["attacker_init"]
        x = self.rng.uniform(*ai["x_range"])
        y = self.rng.uniform(*ai["y_range"])
        heading = self.rng.uniform(*ai["heading_range"])
        self.attacker = Aircraft(0, "attacker", cfg["attacker"], x, y,
                                cfg["attacker"]["v_nominal"], heading)

        self.escorts = []
        for i in range(self.n_escorts):
            ei = cfg["escort_init"]
            x = self.rng.uniform(*ei["x_range"])
            y = self.rng.uniform(*ei["y_range"])
            heading = self.rng.uniform(*ei["heading_range"])
            esc = Aircraft(1 + i, "escort", cfg["escort"], x, y,
                          cfg["escort"]["v_nominal"], heading)
            self.escorts.append(esc)

        self.interceptors = []
        self.interceptor_policies = []
        for i in range(self.n_interceptors):
            ii = cfg["interceptor_init"]
            x = self.rng.uniform(*ii["x_range"])
            y = self.rng.uniform(*ii["y_range"])
            heading = self.rng.uniform(*ii["heading_range"])
            intc = Aircraft(100 + i, "interceptor", cfg["interceptor"], x, y,
                           cfg["interceptor"]["v_nominal"], heading)
            self.interceptors.append(intc)
            policy = InterceptorPolicy(intc, self.hvt, cfg, patrol_idx=i)
            self.interceptor_policies.append(policy)

    def reset(self):
        self.current_step = 0
        self._create_entities()
        for p in self.interceptor_policies:
            p.reset()

        self.prev_dist_to_hvt = self.attacker.distance_to(self.hvt.x, self.hvt.y)
        self.exposure_steps_attacker = 0
        self.cumulative_cost = 0.0
        self.fov_violation_count = 0
        self.intercept_events = []
        self.escort_kill_events = []

        obs = self._get_obs()
        share_obs = self._get_share_obs()
        avail_actions = self._get_avail_actions()

        return obs, share_obs, avail_actions

    def step(self, actions):
        cfg = self.config
        self.current_step += 1
        step_escort_kills = []

        # 1. 训练 agent 执行动作
        all_trained = [self.attacker] + self.escorts
        for i, agent in enumerate(all_trained):
            if agent.alive:
                action = np.array(actions[i], dtype=np.float32)
                agent.step_with_action(action, self.dt)

        # 2. 拦截器执行比例导引
        for i, policy in enumerate(self.interceptor_policies):
            intc = self.interceptors[i]
            if intc.alive:
                nx_cmd, ny_cmd = policy.get_action(self.attacker, self.escorts, self.dt)
                intc.step(nx_cmd, ny_cmd, self.dt)

        # 3. 检查击杀事件 (所有3m距离)
        step_escort_kills = self._check_kills()
        self._check_boundary()

        # 4. 计算奖励
        rewards_list, reward_info = compute_rewards(
            self.attacker, self.escorts, self.interceptors, self.hvt,
            cfg, self.prev_dist_to_hvt, step_escort_kills)

        self.prev_dist_to_hvt = (self.attacker.distance_to(self.hvt.x, self.hvt.y)
                                 if self.attacker.alive else float('inf'))

        # 5. 计算成本
        costs_list, cost_info = compute_costs(
            self.attacker, self.escorts, self.interceptors, cfg)

        # 统计
        if self.attacker.alive:
            for intc in self.interceptors:
                if intc.alive and intc.is_in_fov(self.attacker.x, self.attacker.y,
                                                   cfg["fov_half_angle"], cfg["detection_range"]):
                    self.exposure_steps_attacker += 1
                    self.fov_violation_count += 1
                    break

        self.cumulative_cost += sum(costs_list)

        # 6. 终止判定
        done, done_reason = self._check_done()

        # 超时惩罚
        if done and done_reason == "timeout":
            for i in range(self.n_agents):
                rewards_list[i] += cfg["reward"]["timeout_penalty"]

        # 7. 组装返回
        rewards = [[r] for r in rewards_list]
        costs = [[c] for c in costs_list]
        dones = [done] * self.n_agents

        escorts_alive = sum(1 for e in self.escorts if e.alive)
        intc_alive = sum(1 for i in self.interceptors if i.alive)

        cost_for_info = [[c] for c in costs_list]

        info = {
            "cost": cost_for_info,
            "success": done_reason == "hit_hvt",
            "attacker_killed": not self.attacker.alive,
            "escorts_alive_count": escorts_alive,
            "interceptors_alive_count": intc_alive,
            "exposure_steps_attacker": self.exposure_steps_attacker,
            "cumulative_cost": self.cumulative_cost,
            "hit_target": done_reason == "hit_hvt",
            "min_distance_to_target": (self.attacker.distance_to(self.hvt.x, self.hvt.y)
                                       if self.attacker.alive else float('inf')),
            "fov_violation_count": self.fov_violation_count,
            "intercept_events": list(self.intercept_events),
            "escort_kill_events": list(self.escort_kill_events),
            "done_reason": done_reason,
        }

        if done:
            info["bad_transition"] = (done_reason == "timeout")

        infos = [info for _ in range(self.n_agents)]

        obs = self._get_obs()
        share_obs = self._get_share_obs()
        avail_actions = self._get_avail_actions()

        return obs, share_obs, rewards, costs, dones, infos, avail_actions

    def _check_kills(self):
        """
        检查所有击杀事件 (统一3m击杀距离)

        1. 拦截器 → 进攻/护卫飞行器
        2. 护卫飞行器 → 拦截器
        3. 进攻飞行器 → HVT (在 _check_done 中处理)
        """
        cfg = self.config
        kill_range = cfg["kill_range"]
        escort_kills = []

        # 拦截器击杀进攻/护卫飞行器
        all_trained = [self.attacker] + self.escorts
        for intc in self.interceptors:
            if not intc.alive:
                continue
            for agent in all_trained:
                if not agent.alive:
                    continue
                if intc.distance_to(agent.x, agent.y) < kill_range:
                    agent.kill()
                    self.intercept_events.append({
                        "step": self.current_step,
                        "interceptor_id": intc.uid,
                        "target_role": agent.role,
                        "target_id": agent.uid,
                    })

        # 护卫飞行器击杀拦截器
        for ei, esc in enumerate(self.escorts):
            if not esc.alive:
                continue
            for intc in self.interceptors:
                if not intc.alive:
                    continue
                if esc.distance_to(intc.x, intc.y) < kill_range:
                    intc.kill()
                    kill_event = {
                        "step": self.current_step,
                        "escort_idx": ei,
                        "escort_id": esc.uid,
                        "interceptor_id": intc.uid,
                    }
                    escort_kills.append(kill_event)
                    self.escort_kill_events.append(kill_event)

        return escort_kills

    def _check_boundary(self):
        map_size = self.config["map_size"]
        all_trained = [self.attacker] + self.escorts
        for agent in all_trained:
            if agent.alive:
                if abs(agent.x) > map_size * 1.5 or abs(agent.y) > map_size * 1.5:
                    agent.kill()

    def _check_done(self):
        cfg = self.config
        kill_range = cfg["kill_range"]

        # 进攻飞行器成功打击HVT → 进攻方胜
        if self.attacker.alive and self.attacker.distance_to(self.hvt.x, self.hvt.y) <= kill_range:
            return True, "hit_hvt"

        # 进攻飞行器被拦截 → 防御方胜
        if not self.attacker.alive:
            return True, "attacker_killed"

        # 超时
        if self.current_step >= self.max_steps:
            return True, "timeout"

        return False, ""

    def _get_obs(self):
        cfg = self.config
        obs_range = cfg["obs_range"]
        vel_range = cfg["vel_range"]
        all_trained = [self.attacker] + self.escorts

        # attacker 暴露状态
        attacker_exposed = 0.0
        if self.attacker.alive:
            for intc in self.interceptors:
                if intc.alive and intc.is_in_fov(self.attacker.x, self.attacker.y,
                                                   cfg["fov_half_angle"], cfg["detection_range"]):
                    attacker_exposed = 1.0
                    break

        obs_list = []
        for ai, agent in enumerate(all_trained):
            obs = []

            # 1. 自身状态 (含当前过载)
            obs.extend([
                agent.x / obs_range,
                agent.y / obs_range,
                agent.v / vel_range,
                agent.heading / np.pi,
                agent.nx / 4.0,  # 归一化过载
                agent.ny / 4.0,
            ])

            # 2. HVT 相对位置
            obs.extend([
                (self.hvt.x - agent.x) / obs_range,
                (self.hvt.y - agent.y) / obs_range,
            ])

            # 3. 所有拦截器信息
            for intc in self.interceptors:
                if intc.alive and agent.alive:
                    obs.extend([
                        (intc.x - agent.x) / obs_range,
                        (intc.y - agent.y) / obs_range,
                        (intc.v - agent.v) / vel_range,
                        (intc.heading - agent.heading) / np.pi,
                        1.0,  # alive
                    ])
                else:
                    obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            # 4. FOV 状态
            in_fov = 0.0
            if agent.alive:
                for intc in self.interceptors:
                    if intc.alive and intc.is_in_fov(agent.x, agent.y,
                                                       cfg["fov_half_angle"], cfg["detection_range"]):
                        in_fov = 1.0
                        break
            obs.append(in_fov)
            obs.append(attacker_exposed)

            # 5. 队友信息
            for aj, teammate in enumerate(all_trained):
                if aj == ai:
                    continue
                if teammate.alive and agent.alive:
                    obs.extend([
                        (teammate.x - agent.x) / obs_range,
                        (teammate.y - agent.y) / obs_range,
                        teammate.v / vel_range,
                        teammate.heading / np.pi,
                        1.0,  # alive
                    ])
                else:
                    obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            # 6. 最近威胁距离
            min_threat = 1.0
            if agent.alive:
                for intc in self.interceptors:
                    if intc.alive:
                        d = agent.distance_to(intc.x, intc.y) / obs_range
                        min_threat = min(min_threat, d)
            obs.append(min_threat)

            # 7. 到HVT距离
            dist_hvt = agent.distance_to(self.hvt.x, self.hvt.y) / obs_range if agent.alive else 1.0
            obs.append(dist_hvt)

            obs_list.append(np.array(obs, dtype=np.float32))

        return obs_list

    def _get_share_obs(self):
        cfg = self.config
        obs_range = cfg["obs_range"]
        vel_range = cfg["vel_range"]
        all_trained = [self.attacker] + self.escorts

        so = []

        # 1. 训练agent状态 (含过载)
        for agent in all_trained:
            so.extend([
                agent.x / obs_range,
                agent.y / obs_range,
                agent.v / vel_range,
                agent.heading / np.pi,
                agent.nx / 4.0,
                agent.ny / 4.0,
                float(agent.alive),
            ])

        # 2. 拦截器状态
        for intc in self.interceptors:
            so.extend([
                intc.x / obs_range,
                intc.y / obs_range,
                intc.v / vel_range,
                intc.heading / np.pi,
                float(intc.alive),
            ])

        # 3. HVT
        so.extend([self.hvt.x / obs_range, self.hvt.y / obs_range])

        # 4. 攻防关系
        dist_atk_hvt = (self.attacker.distance_to(self.hvt.x, self.hvt.y) / obs_range
                        if self.attacker.alive else 1.0)

        atk_fov_count = 0
        if self.attacker.alive:
            for intc in self.interceptors:
                if intc.alive and intc.is_in_fov(self.attacker.x, self.attacker.y,
                                                   cfg["fov_half_angle"], cfg["detection_range"]):
                    atk_fov_count += 1

        escorts_alive = sum(1 for e in self.escorts if e.alive) / max(self.n_escorts, 1)
        intc_alive = sum(1 for i in self.interceptors if i.alive) / max(self.n_interceptors, 1)

        so.extend([
            dist_atk_hvt,
            atk_fov_count / max(self.n_interceptors, 1),
            escorts_alive,
            intc_alive,
        ])

        so_array = np.array(so, dtype=np.float32)
        return [so_array.copy() for _ in range(self.n_agents)]

    def _get_avail_actions(self):
        return np.ones((self.n_agents, 2), dtype=np.float32)

    def close(self):
        pass

    def render(self, mode="human"):
        pass

    def get_env_info(self):
        return {
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "share_obs_dim": self.share_obs_dim,
            "action_dim": 2,
            "n_interceptors": self.n_interceptors,
            "map_size": self.config["map_size"],
        }
