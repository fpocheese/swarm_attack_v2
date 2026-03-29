"""
FOV Penetration Environment - Interceptor Policy V21
====================================================
三维比例导引法 (3D PN) + 持续追击

V21 拦截器策略 (2026-03-28):
    1. 拦截器对已分配目标持续追击, 首次逃逸后不再放弃
    2. 探测区内高频信息更新(20Hz), 探测区外低频信息更新(2Hz)
    3. 保留过载/逃逸事件记录, 但不因事件而清空目标分配
    4. 默认允许回头追击, 满足"无论如何都去打击自己的目标"
    5. 修复外推定时器bug: guide分支更新后也重置info_timer
    6. 首次获取目标时立即初始化已知位置, 确保PN导引从分配瞬间生效
    7. 启用周期性重分配+全覆盖保证, 每个进攻方都有拦截器
"""

import numpy as np
from .config import G


class InterceptorPolicy:
    """基于 3D PN 的拦截器控制器 — V21 持续追击 + 全覆盖分配"""

    # 交战状态
    STATE_PATROL = 0       # 巡逻
    STATE_APPROACH = 1     # 接近中(远距离)
    STATE_ENGAGED = 2      # 交战中(进入engagement_range)
    STATE_MISSED = 3       # 已错失(目标逃脱或飞过)
    STATE_ABANDONED = 4    # 已放弃(目标飞到身后)

    def __init__(self, interceptor, hvt, config, patrol_idx=0):
        self.interceptor = interceptor
        self.hvt = hvt
        self.config = config
        self.patrol_idx = patrol_idx
        self.N = config["pn_nav_gain"]
        self.direct_freq = config["pn_direct_freq"]
        self.guide_freq = config["pn_guide_freq"]
        self.assigned_target_idx = None
        self.target = None
        self.target_pos_known = None
        self.prev_los_az = None
        self.prev_los_el = None
        self.info_timer = 0.0
        self.guide_timer = 0.0

        # 交战状态跟踪
        self.engagement_state = self.STATE_PATROL
        self.tracking_steps = 0           # 连续FOV锁定步数
        self.demanded_ny = 0.0            # 上一步要求的ny (供外部读取)
        self.demanded_nz = 0.0            # 上一步要求的nz
        self.closing_speed = 0.0          # 闭合速度
        self.los_rate_az = 0.0            # 视线角速度(方位)
        self.los_rate_el = 0.0            # 视线角速度(俯仰)
        self.engagement_min_dist = float('inf')  # 交战过程中最近距离

        # 巡逻偏移
        n_def = config["n_defensive"]
        angle = 2 * np.pi * patrol_idx / max(n_def, 1) + np.pi
        self.patrol_offset_x = 800.0 * np.cos(angle)
        self.patrol_offset_y = 800.0 * np.sin(angle)

        # 追踪配置
        self.pursuit_cfg = config.get("pursuit", {})
        self.fov_escape_cfg = config.get("fov_escape", {})

    def reset(self):
        self.assigned_target_idx = None
        self.target = None
        self.target_pos_known = None
        self.prev_los_az = None
        self.prev_los_el = None
        self.info_timer = 0.0
        self.guide_timer = 0.0
        self.engagement_state = self.STATE_PATROL
        self.tracking_steps = 0
        self.demanded_ny = 0.0
        self.demanded_nz = 0.0
        self.closing_speed = 0.0
        self.los_rate_az = 0.0
        self.los_rate_el = 0.0
        self.engagement_min_dist = float('inf')

    def set_target(self, target_idx, target_aircraft):
        if self.assigned_target_idx != target_idx:
            self.prev_los_az = None
            self.prev_los_el = None
            self.tracking_steps = 0
            self.engagement_min_dist = float('inf')
            if self.engagement_state != self.STATE_MISSED:
                self.engagement_state = self.STATE_APPROACH
            # V21: 立即获取新目标的位置信息, 确保PN导引从分配瞬间开始
            if target_aircraft is not None and target_aircraft.alive:
                cos_g = np.cos(target_aircraft.gamma)
                self.target_pos_known = [
                    target_aircraft.x, target_aircraft.y, target_aircraft.z,
                    target_aircraft.v * cos_g * np.cos(target_aircraft.heading),
                    target_aircraft.v * cos_g * np.sin(target_aircraft.heading),
                    target_aircraft.v * np.sin(target_aircraft.gamma),
                ]
                self.info_timer = 0.0
                self.guide_timer = 0.0
            else:
                self.target_pos_known = None
        self.assigned_target_idx = target_idx
        self.target = target_aircraft

    def mark_target_missed(self, target_idx):
        """记录一次脱靶/逃逸事件, 但继续追击原分配目标。"""
        if self.assigned_target_idx == target_idx:
            self.engagement_state = self.STATE_APPROACH
            self.tracking_steps = 0
            self.prev_los_az = None
            self.prev_los_el = None

    def get_action(self, offensives, dt):
        intc = self.interceptor
        if not intc.alive:
            return 0.0, 0.0, 0.0

        fov_half = self.config["fov_half_angle"]
        det_range = self.config["detection_range"]
        self.info_timer += dt
        self.guide_timer += dt

        # 确定追踪目标
        target = self._resolve_target(offensives)
        if target is None:
            self.demanded_ny = 0.0
            self.demanded_nz = 0.0
            self.closing_speed = 0.0
            return self._patrol_action()

        # V21: 首次获取目标或逃逸重置后, 立即初始化已知位置
        #      确保拦截器从第一步就使用PN导引追击目标
        if self.target_pos_known is None:
            cos_g = np.cos(target.gamma)
            self.target_pos_known = [
                target.x, target.y, target.z,
                target.v * cos_g * np.cos(target.heading),
                target.v * cos_g * np.sin(target.heading),
                target.v * np.sin(target.gamma),
            ]
            self.info_timer = 0.0
            self.guide_timer = 0.0

        # 更新交战状态
        dist = intc.distance_3d(target)
        engagement_range = self.fov_escape_cfg.get("engagement_range", 200.0)
        if dist < engagement_range:
            self.engagement_state = self.STATE_ENGAGED
        self.engagement_min_dist = min(self.engagement_min_dist, dist)

        # 检查目标是否在FOV中
        target_in_fov = (target.alive and
                         intc.is_in_fov(target.x, target.y, target.z,
                                        fov_half, det_range))

        # 更新FOV跟踪计数
        if target_in_fov:
            self.tracking_steps += 1
        else:
            self.tracking_steps = 0

        # 信息更新
        #   FOV内: 高频(20Hz)直接测量 → 精确PN导引
        #   FOV外: 低频(2Hz)引导信息  → 外推PN导引
        #   V21修复: 两个分支都重置info_timer, 确保外推时间正确
        if target_in_fov and self.info_timer >= 1.0 / self.direct_freq:
            cos_g = np.cos(target.gamma)
            self.target_pos_known = [
                target.x, target.y, target.z,
                target.v * cos_g * np.cos(target.heading),
                target.v * cos_g * np.sin(target.heading),
                target.v * np.sin(target.gamma),
            ]
            self.info_timer = 0.0
            self.guide_timer = 0.0
        elif self.guide_timer >= 1.0 / self.guide_freq:
            cos_g = np.cos(target.gamma)
            self.target_pos_known = [
                target.x, target.y, target.z,
                target.v * cos_g * np.cos(target.heading),
                target.v * cos_g * np.sin(target.heading),
                target.v * np.sin(target.gamma),
            ]
            self.info_timer = 0.0   # V21修复: guide更新后也重置外推定时器
            self.guide_timer = 0.0

        # 外推目标位置
        tx = self.target_pos_known[0] + self.target_pos_known[3] * self.info_timer
        ty = self.target_pos_known[1] + self.target_pos_known[4] * self.info_timer
        tz = self.target_pos_known[2] + self.target_pos_known[5] * self.info_timer

        return self._pn_guidance_3d(tx, ty, tz, dt)

    def _resolve_target(self, offensives):
        """目标解析 — V21: 只要分配目标还活着, 就持续追击。"""

        if self.target is None or not self.target.alive:
            return None

        intc = self.interceptor
        target = self.target

        # 可选前向追踪限制；V21默认关闭以允许回头继续追击既定目标
        if self.pursuit_cfg.get("forward_only", False):
            dx = target.x - intc.x
            dy = target.y - intc.y
            dz = target.z - intc.z
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            if dist > 1.0:
                # 计算目标相对于拦截器航向的偏角
                cos_g = np.cos(intc.gamma)
                fwd_x = cos_g * np.cos(intc.heading)
                fwd_y = cos_g * np.sin(intc.heading)
                fwd_z = np.sin(intc.gamma)
                # 目标方向单位向量
                tgt_x, tgt_y, tgt_z = dx / dist, dy / dist, dz / dist
                # 点积 = cos(偏角)
                dot = fwd_x * tgt_x + fwd_y * tgt_y + fwd_z * tgt_z
                off_axis = np.arccos(np.clip(dot, -1.0, 1.0))

                forward_half = self.pursuit_cfg.get("forward_half_angle", np.deg2rad(100.0))
                if off_axis > forward_half:
                    if self.pursuit_cfg.get("abandon_on_pass", True):
                        self.engagement_state = self.STATE_ABANDONED
                        return None

        return target

    def _pn_guidance_3d(self, tx, ty, tz, dt):
        """3D比例导引 — 记录demanded overload供外部检查"""
        intc = self.interceptor
        dx = tx - intc.x
        dy = ty - intc.y
        dz = tz - intc.z
        r = max(np.sqrt(dx**2 + dy**2 + dz**2), 1.0)

        los_az = np.arctan2(dy, dx)
        r_horiz = max(np.sqrt(dx**2 + dy**2), 1.0)
        los_el = np.arctan2(dz, r_horiz)

        if self.prev_los_az is not None:
            d_az = np.arctan2(np.sin(los_az - self.prev_los_az),
                              np.cos(los_az - self.prev_los_az))
            los_rate_az = d_az / dt
        else:
            los_rate_az = 0.0

        if self.prev_los_el is not None:
            d_el = np.arctan2(np.sin(los_el - self.prev_los_el),
                              np.cos(los_el - self.prev_los_el))
            los_rate_el = d_el / dt
        else:
            los_rate_el = 0.0

        self.prev_los_az = los_az
        self.prev_los_el = los_el
        self.los_rate_az = los_rate_az
        self.los_rate_el = los_rate_el

        cos_g = np.cos(intc.gamma)
        vx_i = intc.v * cos_g * np.cos(intc.heading)
        vy_i = intc.v * cos_g * np.sin(intc.heading)
        vz_i = intc.v * np.sin(intc.gamma)
        v_closing = max(-(dx * vx_i + dy * vy_i + dz * vz_i) / r, 10.0)
        self.closing_speed = v_closing

        # PN导引律需求的过载 (记录未裁剪值, 供外部检查是否过载)
        ny_demanded = self.N * v_closing * los_rate_az / G
        nz_demanded = self.N * v_closing * los_rate_el / G + np.cos(intc.gamma)
        nx_cmd = 0.5

        # 保存demanded值(未裁剪) — 外部可用此检测过载饱和
        self.demanded_ny = ny_demanded
        self.demanded_nz = nz_demanded

        # 裁剪到拦截器实际能力范围
        params = intc.params
        nx_cmd = np.clip(nx_cmd, params["nx_min"], params["nx_max"])
        ny_cmd = np.clip(ny_demanded, params["ny_min"], params["ny_max"])
        nz_cmd = np.clip(nz_demanded, params.get("nz_min", -3.0), params.get("nz_max", 3.0))
        return nx_cmd, ny_cmd, nz_cmd

    def _patrol_action(self):
        intc = self.interceptor
        px = self.hvt.x + self.patrol_offset_x
        py = self.hvt.y + self.patrol_offset_y
        pz = self.hvt.z + 500.0
        dx = px - intc.x
        dy = py - intc.y
        heading_err = np.arctan2(dy, dx) - intc.heading
        heading_err = np.arctan2(np.sin(heading_err), np.cos(heading_err))
        ny_cmd = np.clip(2.0 * heading_err, -2.0, 2.0)
        dz = pz - intc.z
        nz_cmd = np.clip(0.5 * dz / 100.0, -1.0, 1.0) + np.cos(intc.gamma)
        return 0.0, ny_cmd, nz_cmd

    def _goto_hvt(self):
        intc = self.interceptor
        dx = self.hvt.x - intc.x
        dy = self.hvt.y - intc.y
        heading_err = np.arctan2(dy, dx) - intc.heading
        heading_err = np.arctan2(np.sin(heading_err), np.cos(heading_err))
        ny_cmd = np.clip(3.0 * heading_err, -4.0, 4.0)
        dz = (self.hvt.z + 500.0) - intc.z
        nz_cmd = np.clip(0.5 * dz / 100.0, -1.0, 1.0) + np.cos(intc.gamma)
        return 0.5, ny_cmd, nz_cmd

    def is_overload_saturated(self):
        """
        检查当前PN要求的过载是否超出拦截器极限
        返回: (saturated_ny, saturated_nz, saturation_ratio)
        saturation_ratio = max(|demanded|/max_available) — 越大说明越跟不住
        """
        params = self.interceptor.params
        ny_max = params["ny_max"]
        nz_max = params.get("nz_max", 3.0)
        ny_ratio = abs(self.demanded_ny) / max(ny_max, 0.1)
        nz_ratio = abs(self.demanded_nz) / max(nz_max, 0.1)
        saturated_ny = abs(self.demanded_ny) > ny_max
        saturated_nz = abs(self.demanded_nz) > nz_max
        return saturated_ny, saturated_nz, max(ny_ratio, nz_ratio)
