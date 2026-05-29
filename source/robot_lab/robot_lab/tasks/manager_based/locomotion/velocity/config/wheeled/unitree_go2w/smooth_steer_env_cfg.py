# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .flat_env_cfg import UnitreeGo2WFlatEnvCfg


@configclass
class UnitreeGo2WSmoothSteerEnvCfg(UnitreeGo2WFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity = mdp.UniformDifferentialCommandCfg(
            resampling_time_range=(10.0, 10.0),
            ranges=mdp.UniformDifferentialCommandCfg.Ranges(
                lin_acc_x=(-2.0, 2.0),
                roll_rate=(0.0, 0.0),      # always command zero roll — flat chassis
                ang_vel_z=(-math.pi / 3, math.pi / 3),
            ),
        )

        # disable old velocity tracker (no longer the goal)
        self.rewards.track_lin_vel_xy_exp.weight = 0.0

        # longitudinal acceleration tracking (cmd index 0)
        setattr(self.rewards, "track_lin_acc_x_exp", RewTerm(
            func=mdp.track_lin_acc_x_exp,
            weight=2.0,
            params={"command_name": "base_velocity", "std": 1.0},
        ))

        # roll rate tracking (cmd index 1 — always 0, so this penalizes any roll)
        setattr(self.rewards, "track_roll_rate_exp", RewTerm(
            func=mdp.track_roll_rate_exp,
            weight=1.5,
            params={"command_name": "base_velocity", "std": 0.3},
        ))

        self.rewards.track_ang_vel_z_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5

        # level body (roll + pitch angle)
        self.rewards.flat_orientation_l2.weight = -5.0

        # maintain 0.40m body height (target_height already set in rough_env_cfg)
        self.rewards.base_height_l2.weight = -3.0

        # strong default-pose hold — applies during motion too
        self.rewards.joint_pos_penalty.weight = -8.0

        # penalize any leg joint velocity (no stepping)
        self.rewards.joint_vel_l2.weight = -0.05

        # all 4 hips equal, all 4 thighs equal, all 4 calves equal
        self.rewards.action_sync.weight = -1.0

        # roll/pitch rate (complements track_roll_rate_exp — also catches pitch)
        self.rewards.ang_vel_xy_l2.weight = -0.3

        # wheel jerk
        self.rewards.joint_acc_wheel_l2.weight = -5e-8

        # wheel torque magnitude
        self.rewards.joint_torques_wheel_l2.weight = -5e-4

        # action smoothness
        self.rewards.action_rate_l2.weight = -0.05

        self.scene.terrain.physics_material.static_friction = 1.1
        self.scene.terrain.physics_material.dynamic_friction = 1.1
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.8, 1.2)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.7, 1.1)

        if self.__class__.__name__ == "UnitreeGo2WSmoothSteerEnvCfg":
            self.disable_zero_weight_rewards()
