# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Paper-faithful moment-steering policy for the Unitree Go2-W.

The low-level RL policy observes the MPC command ``[a_x, M_psi, M_theta]`` (longitudinal
acceleration, yaw moment, roll moment) and produces 16-d joint targets that drive the robot
around a track while *generating body roll itself* — no external joint overlay. See the design
plan for the full rationale; the key idea is that the command is integrated into reference body
states (v*, wz*, theta*) and the reward tracks those, so a sustained roll moment yields a held
bank with a real gradient (unlike commanding a roll rate, which is zero at a held lean).

Observation layout (order set by the base PolicyCfg, 56-d total):
    base_lin_vel(3) | base_ang_vel(3) | projected_gravity(3) | command[a_x,M_psi,M_theta](3)
    | leg_joint_pos_rel(12) | all_motor_vel(16) | prev_action(16)
This ordering is the deployment contract frozen into the exported policy.
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import UnitreeGo2WRoughEnvCfg


@configclass
class UnitreeGo2WMomentSteerEnvCfg(UnitreeGo2WRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Scene (flat plane, keep contact sensor)------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.curriculum.terrain_levels = None
        # nominal high-grip planar track; keep the paper's friction randomization (inherited)
        self.scene.terrain.physics_material.static_friction = 1.0
        self.scene.terrain.physics_material.dynamic_friction = 1.0

        # ------------------------------Observations (56-d)------------------------------
        # base_lin_vel, base_ang_vel, projected_gravity, velocity_commands all kept from Rough.
        # Trim joint_pos to the 12 leg joints only (paper: q_l in R^12), no wheel-zeroing variant.
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_delayed
        self.observations.policy.joint_pos.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names, preserve_order=True)
        }
        self.observations.policy.joint_pos.scale = 1.0
        # joint_vel stays all 16 motors (set in Rough), scale 0.05.
        # drop height scan
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None

        # ------------------------------Commands------------------------------
        # Replace the velocity command with the paper's [a_x, M_psi, M_theta] moment command.
        # Keep the name "base_velocity" so existing command_name params / generated_commands wiring works.
        self.commands.base_velocity = mdp.UniformMomentCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.5, 4.0),
            rel_standing_envs=0.05,
            ranges=mdp.UniformMomentCommandCfg.Ranges(
                lin_accel_x=(-2.0, 2.0),
                yaw_moment=(-8.0, 8.0),
                roll_moment=(-15.0, 15.0),
                roll_moment_noise=(-5.0, 5.0),
            ),
        )

        # ------------------------------Rewards------------------------------
        # Setpoint-tracking (integrated from the moment command)
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        setattr(self.rewards, "track_setpoint_lin_vel_exp", RewTerm(
            func=mdp.track_setpoint_lin_vel_exp,
            weight=3.0,
            params={"command_name": "base_velocity", "std": 0.25},
        ))
        setattr(self.rewards, "track_setpoint_yaw_rate_exp", RewTerm(
            func=mdp.track_setpoint_yaw_rate_exp,
            weight=1.5,
            params={"command_name": "base_velocity", "std": 0.30},
        ))
        setattr(self.rewards, "track_setpoint_roll_exp", RewTerm(
            func=mdp.track_setpoint_roll_exp,
            weight=3.0,  # Tier-1 boost so the lean dominates residual posture penalties
            params={"command_name": "base_velocity", "std": 0.10},
        ))

        # Tier 1: decouple roll from posture — pitch-only orientation + pitch-rate, drop symmetric terms
        self.rewards.upward.weight = 0.0
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0
        setattr(self.rewards, "flat_orientation_pitch_l2", RewTerm(
            func=mdp.flat_orientation_pitch_l2,
            weight=-5.0,
        ))
        setattr(self.rewards, "ang_vel_y_l2", RewTerm(
            func=mdp.ang_vel_y_l2,
            weight=-0.3,
        ))
        # Tier 1: lean-permitting symmetry — mirror front<->rear per side (blocks gait/pitch
        # asymmetry, allows left/right roll). Replaces the diagonal mirror that fights the lean.
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR.*", "RR.*"],  # right side: front+rear mirror
            ["FL.*", "RL.*"],  # left side: front+rear mirror
        ]

        # Tier 4: hold constant chassis height (Assumption 1, constant CG height h=0.40)
        self.rewards.base_height_l2.weight = -3.0
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.base_height_l2.params["sensor_cfg"] = None  # flat plane, absolute height

        # Paper penalties (Table rl_params)
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_power.weight = -2.0e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.undesired_contacts.weight = -12.0  # non-foot body contact (body_names set in Rough)
        self.rewards.contact_forces.weight = -1.5e-4  # foot contact force magnitude

        # Keep joint position limits as a safety guard; zero terms outside our set.
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_penalty.weight = 0.0  # optional default-pose hold (enable if legs drift)
        self.rewards.stand_still.weight = 0.0
        self.rewards.wheel_vel_penalty.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.joint_acc_l2.weight = 0.0
        self.rewards.joint_acc_wheel_l2.weight = 0.0
        self.rewards.joint_torques_wheel_l2.weight = 0.0
        self.rewards.joint_vel_l2.weight = 0.0
        self.rewards.joint_vel_wheel_l2.weight = 0.0

        # ------------------------------Curriculum------------------------------
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        setattr(self.curriculum, "command_levels_moment", CurrTerm(
            func=mdp.command_levels_moment,
            params={"reward_term_name": "track_setpoint_yaw_rate_exp", "range_multiplier": (0.2, 1.0)},
        ))

        # ------------------------------Events (start upright like a vehicle)------------------------------
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.0, 0.1),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-3.14, 3.14),
        }

        # ------------------------------Terminations------------------------------
        # End the episode if the chassis itself contacts the ground (rollover / fall).
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.base_link_name]),
                "threshold": 1.0,
            },
        )

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2WMomentSteerEnvCfg":
            self.disable_zero_weight_rewards()
