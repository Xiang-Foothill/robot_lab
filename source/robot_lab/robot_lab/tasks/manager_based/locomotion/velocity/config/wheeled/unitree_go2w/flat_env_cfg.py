# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity import mdp

from .rough_env_cfg import UnitreeGo2WRoughEnvCfg


@configclass
class UnitreeGo2WFlatEnvCfg(UnitreeGo2WRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # remove gravity from observation
        self.observations.policy.projected_gravity = None
        self.observations.critic.projected_gravity = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        
        # ------------------------------Commands------------------------------
        """The commands given during training and evaluation.
        Override the default command ranges for the purpose of high-speed locomotion."""
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 5.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi, math.pi)
        
        # reset the base pose and velocity randomly during reset
        #TODO: figure out the physics meaning of roll, pitch yaw here. The current values does not make sense.
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-1.047, 1.047),  # -60 to 60 degrees
                "pitch": (-1.047, 1.047),  # -60 to 60 degrees
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        # ------------------------------Curriculums------------------------------
        # remove curriculum terms related to terrain and height
        
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2WFlatEnvCfg":
            self.disable_zero_weight_rewards()
