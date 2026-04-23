# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs import ManagerBasedRLEnv

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp


class ManagerBasedRLEnvWithObsDelay(ManagerBasedRLEnv):
    """Manager-based RL env with optional physics-step observation delay recording.

    The delay buffers are updated on each call to ``_apply_action``. In environments where
    ``_apply_action`` is invoked per physics sub-step, this provides physics-step granularity.
    """

    def _apply_action(self):
        super()._apply_action()
        mdp.record_observation_delay_sample(self, force=True)
