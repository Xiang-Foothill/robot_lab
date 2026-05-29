# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""Deploy-side adapter for the Go2W smooth-steer policy (54-d obs -> 16-d action)."""

from __future__ import annotations

import numpy as np
import torch

LEG_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
WHEEL_JOINTS = ["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"]
JOINT_ORDER = LEG_JOINTS + WHEEL_JOINTS

DEFAULT_Q = np.array([0.0, 0.8, -1.5] * 4 + [0.0, 0.0, 0.0, 0.0], dtype=np.float64)
LEG_SCALE = np.array([0.125 if "hip" in j else 0.25 for j in LEG_JOINTS], dtype=np.float64)
WHEEL_VEL_SCALE = 5.0
ANG_VEL_SCALE = 0.25
JOINT_POS_SCALE = 1.0
JOINT_VEL_SCALE = 0.05


class PolicyRunner:
    def __init__(self, policy_path, device="cpu"):
        self.device = device
        self.policy = torch.jit.load(policy_path, map_location=device).eval()
        self.last_action = np.zeros(16, dtype=np.float64)
        self.n_legs = len(LEG_JOINTS)

    def reset(self):
        self.last_action[:] = 0.0

    def build_obs(self, base_ang_vel, command, joint_pos, joint_vel):
        joint_pos_rel = np.asarray(joint_pos, dtype=np.float64) - DEFAULT_Q
        joint_pos_rel[self.n_legs:] = 0.0
        obs = np.concatenate([
            np.asarray(base_ang_vel, dtype=np.float64) * ANG_VEL_SCALE,
            np.asarray(command, dtype=np.float64),
            joint_pos_rel * JOINT_POS_SCALE,
            np.asarray(joint_vel, dtype=np.float64) * JOINT_VEL_SCALE,
            self.last_action,
        ])
        return obs.astype(np.float32)

    @torch.no_grad()
    def act(self, obs):
        t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(self.device)
        a = self.policy(t).squeeze(0).cpu().numpy().astype(np.float64)
        self.last_action = a.copy()
        return a

    @staticmethod
    def decode(action):
        a = np.asarray(action, dtype=np.float64)
        leg_q_des = DEFAULT_Q[:12] + LEG_SCALE * a[:12]
        wheel_dq_des = WHEEL_VEL_SCALE * a[12:]
        return leg_q_des, wheel_dq_des
