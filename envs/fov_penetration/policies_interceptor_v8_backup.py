"""
FOV Penetration Environment - Interceptor Policy V3
=====================================================
三维比例导引法 (3D PN) + 目标分配接口
"""

import numpy as np
from .config import G


class InterceptorPolicy:
    """基于 3D 比例导引的拦截器控制器"""

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
        n_def = config["n_defensive"]
        angle = 2 * np.pi * patrol_idx / max(n_def, 1) + np.pi
        self.patrol_offset_x = 800.0 * np.cos(angle)
        self.patrol_offset_y = 800.0 * np.sin(angle)

    def reset(self):
        self.assigned_target_idx = None
        self.target = None
        self.target_pos_known = None
        self.prev_los_az = None
        self.prev_los_el = None
        self.info_timer = 0.0
        self.guide_timer = 0.0

    def set_target(self, target_idx, target_aircraft):
        if self.assigned_target_idx != target_idx:
            self.prev_los_az = None
            self.prev_los_el = None
            self.target_pos_known = None
        self.assigned_target_idx = target_idx
        self.target = target_aircraft

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
            return self._patrol_action()

        # 信息更新
        target_in_fov = (target.alive and
                         intc.is_in_fov(target.x, target.y, target.z,
                                        fov_half, det_range))
        if target_in_fov and self.info_timer >= 1.0 / self.direct_freq:
            cos_g = np.cos(target.gamma)
            self.target_pos_known = [
                target.x, target.y, target.z,
                target.v * cos_g * np.cos(target.heading),
                target.v * cos_g * np.sin(target.heading),
                target.v * np.sin(target.gamma),
            ]
            self.info_timer = 0.0
        elif self.guide_timer >= 1.0 / self.guide_freq:
            cos_g = np.cos(target.gamma)
            self.target_pos_known = [
                target.x, target.y, target.z,
                target.v * cos_g * np.cos(target.heading),
                target.v * cos_g * np.sin(target.heading),
                target.v * np.sin(target.gamma),
            ]
            self.guide_timer = 0.0

        if self.target_pos_known is None:
            return self._goto_hvt()

        # 外推目标位置
        tx = self.target_pos_known[0] + self.target_pos_known[3] * self.info_timer
        ty = self.target_pos_known[1] + self.target_pos_known[4] * self.info_timer
        tz = self.target_pos_known[2] + self.target_pos_known[5] * self.info_timer

        return self._pn_guidance_3d(tx, ty, tz, dt)

    def _resolve_target(self, offensives):
        if self.target is not None and self.target.alive:
            return self.target
        intc = self.interceptor
        best = None
        best_dist = float('inf')
        for off in offensives:
            if off.alive and not off.hit_hvt:
                d = intc.distance_3d(off)
                if d < best_dist:
                    best_dist = d
                    best = off
        self.target = best
        return best

    def _pn_guidance_3d(self, tx, ty, tz, dt):
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

        cos_g = np.cos(intc.gamma)
        vx_i = intc.v * cos_g * np.cos(intc.heading)
        vy_i = intc.v * cos_g * np.sin(intc.heading)
        vz_i = intc.v * np.sin(intc.gamma)
        v_closing = max(-(dx * vx_i + dy * vy_i + dz * vz_i) / r, 10.0)

        ny_cmd = self.N * v_closing * los_rate_az / G
        nz_cmd = self.N * v_closing * los_rate_el / G + np.cos(intc.gamma)
        nx_cmd = 0.5

        params = intc.params
        nx_cmd = np.clip(nx_cmd, params["nx_min"], params["nx_max"])
        ny_cmd = np.clip(ny_cmd, params["ny_min"], params["ny_max"])
        nz_cmd = np.clip(nz_cmd, params.get("nz_min", -3.0), params.get("nz_max", 3.0))
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
