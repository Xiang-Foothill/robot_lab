# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .smooth_steer_env_cfg import UnitreeGo2WSmoothSteerEnvCfg


@configclass
class UnitreeGo2WTiltEnvCfg(UnitreeGo2WSmoothSteerEnvCfg):
    """Active-tilt policy: identical to smooth-steer except the robot is allowed (and
    rewarded) to bank into corners. Kept as a separate task so the smooth-steer policy
    stays frozen as the no-tilt control group for the tilt-vs-no-tilt comparison."""

    def __post_init__(self):
        super().__post_init__()

        # open the roll-rate command channel (was pinned to 0 in the no-tilt control group)
        self.commands.base_velocity.ranges.roll_rate = (-1.5, 1.5)

        # allow roll, keep pitch level: swap the roll+pitch orientation penalty for pitch-only
        self.rewards.flat_orientation_l2.weight = 0.0
        setattr(self.rewards, "flat_orientation_pitch_l2", RewTerm(
            func=mdp.flat_orientation_pitch_l2,
            weight=-5.0,
        ))

        # allow roll rate, keep pitch-rate damping: swap roll+pitch rate penalty for pitch-only
        self.rewards.ang_vel_xy_l2.weight = 0.0
        setattr(self.rewards, "ang_vel_y_l2", RewTerm(
            func=mdp.ang_vel_y_l2,
            weight=-0.3,
        ))

        # track_roll_rate_exp drives the lean — boost its weight so it can overcome the
        # joint-pos and action-sync penalties that fight asymmetric leg extension.
        self.rewards.track_roll_rate_exp.weight = 4.0

        # leaning via shoulder (hip) joints requires left-right hip asymmetry, but the
        # thighs and calves should stay 4-way symmetric to preserve the rigid chassis style.
        # resplit action_sync groups: hips → two side-pairs (FR=RR, FL=RL) so lean is free;
        # thighs and calves keep all-4 sync so no gait-like motion is introduced.
        self.rewards.action_sync.weight = -1.0
        self.rewards.action_sync.params["joint_groups"] = [
            ["FR_hip_joint", "RR_hip_joint"],   # right side: front+rear move together
            ["FL_hip_joint", "RL_hip_joint"],   # left side: front+rear move together
            ["FR_thigh_joint", "FL_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
            ["FR_calf_joint", "FL_calf_joint", "RL_calf_joint", "RR_calf_joint"],
        ]

        # joint_pos_penalty fights hip deviation from default (symmetric) pose — reduce it
        # so the hip lean can overcome it, but keep some to maintain chassis posture.
        self.rewards.joint_pos_penalty.weight = -2.0

        if self.__class__.__name__ == "UnitreeGo2WTiltEnvCfg":
            self.disable_zero_weight_rewards()
