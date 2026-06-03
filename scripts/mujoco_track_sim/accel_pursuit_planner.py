# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""Velocity pure-pursuit planner emitting [lin_vel_x, roll_rate, ang_vel_z]."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.spatial import KDTree

HORIZON = 15
DT = 0.1
LOOKAHEAD_M = 0.6
KP_HDG = 2.0
V_STRAIGHT = 2.8
V_CORNER = 0.8
BLEND_KAPPA = 0.5
BLEND_MARGIN = 0.9
WZ_MAX = np.pi / 3
V_MAX = 1.5
V_MIN = 0.1
CORNER_WZ_FRAC = 0.7
BRAKE_LOOKAHEAD_M = 0.9


class _SpeedProfile:
    def __init__(self, ss, theta, v_straight=V_STRAIGHT, v_corner=V_CORNER, blend_margin=BLEND_MARGIN):
        N = len(ss)
        L = float(ss[-1])
        dtheta = np.abs(np.diff(np.unwrap(theta)))
        ds_arr = np.diff(ss)
        kappa = np.concatenate([dtheta / (ds_arr + 1e-9), [0.0]])
        blend = np.clip(kappa / BLEND_KAPPA, 0.0, 1.0)
        v_raw = v_straight - (v_straight - v_corner) * blend
        n_pre = max(1, int(blend_margin / (L / N)))
        ext = np.tile(v_raw, 3)
        self._v = minimum_filter1d(ext, size=2 * n_pre + 1)[N:2 * N]
        self._ss, self._L, self._N = ss, L, N

    def __call__(self, s):
        s = np.asarray(s) % self._L
        idx = np.searchsorted(self._ss, s) % self._N
        return self._v[idx]


class _RacelineProjector:
    def __init__(self, pts, ss, theta):
        self._pts, self._ss, self._theta = pts, ss, theta
        self._L = float(ss[-1])
        self._N = len(pts)
        self._tree = KDTree(pts)

    def project(self, x, y):
        _, idx = self._tree.query([x, y])
        return float(self._ss[idx])

    def at_s(self, s):
        s = np.asarray(s) % self._L
        idx = np.searchsorted(self._ss, s) % self._N
        return self._pts[idx, 0], self._pts[idx, 1], self._theta[idx]


class AccelPursuitPlanner:
    def __init__(self, raceline_npz, lookahead_m=LOOKAHEAD_M, v_straight=V_STRAIGHT,
                 v_corner=V_CORNER, use_raceline_v=True, v_max=V_MAX):
        data = np.load(raceline_npz)
        pts = data["pts"].astype(float)
        ss = data["ss"].astype(float)
        theta = data["theta"].astype(float)
        self._proj = _RacelineProjector(pts, ss, theta)
        if use_raceline_v and "v" in data.files:
            v_arr = data["v"].astype(float)
            L, N = float(ss[-1]), len(ss)
            self._speed = lambda s: v_arr[np.searchsorted(ss, np.asarray(s) % L) % N]
        else:
            self._speed = _SpeedProfile(ss, theta, v_straight=v_straight, v_corner=v_corner)
        self.lookahead = lookahead_m
        self.v_max = v_max
        self.track_length = float(ss[-1])
        self._ss = ss
        self._N = len(ss)
        kappa = np.abs(np.gradient(np.unwrap(theta)) / (np.gradient(ss) + 1e-9))
        v_curve = CORNER_WZ_FRAC * WZ_MAX / (kappa + 1e-9)
        n_pre = max(1, int(BRAKE_LOOKAHEAD_M / (self.track_length / self._N)))
        v_curve = minimum_filter1d(np.tile(v_curve, 3), size=2 * n_pre + 1)[self._N:2 * self._N]
        self._v_curve = np.clip(v_curve, V_MIN, v_max)

    def _curve_speed(self, s):
        idx = np.searchsorted(self._ss, np.asarray(s) % self.track_length) % self._N
        return self._v_curve[idx]

    def plan(self, x, y, psi, v, phi=0.0):
        seq, info = self.plan_horizon(x, y, psi, v)
        return seq[0], info

    def plan_horizon(self, x, y, psi, v):
        s_now = self._proj.project(x, y)
        v_now = max(float(v), 0.1)
        s_steps = s_now + np.arange(HORIZON) * v_now * DT
        lx, ly, _ = self._proj.at_s(s_steps + self.lookahead)
        dx, dy = lx - x, ly - y
        dist = np.maximum(np.hypot(dx, dy), 1e-3)
        alpha = ((np.arctan2(dy, dx) - psi) + np.pi) % (2 * np.pi) - np.pi
        v_ref = np.minimum(self._speed(s_steps), self.v_max)
        v_ref = np.minimum(v_ref, self._curve_speed(s_steps))
        wz = np.clip(2.0 * np.maximum(v_ref, 0.1) * np.sin(alpha) / dist + KP_HDG * alpha,
                     -WZ_MAX, WZ_MAX)
        vx = np.clip(v_ref, V_MIN, self.v_max)
        roll = np.zeros(HORIZON)
        seq = np.stack([vx, roll, wz], axis=1).astype(np.float32)
        rx, ry, rpsi = self._proj.at_s(s_steps)
        info = {"s": s_now, "v_ref": float(v_ref[0]), "v_now": v_now,
                "pred_x": rx, "pred_y": ry, "pred_psi": rpsi}
        return seq, info
