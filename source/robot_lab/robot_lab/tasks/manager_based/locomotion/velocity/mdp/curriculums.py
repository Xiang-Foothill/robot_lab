# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_moment(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.2, 1.0),
    num_steps: int = 8,
) -> torch.Tensor:
    """Curriculum that gradually expands the yaw-moment and roll-moment command ranges.

    Mirrors :func:`command_levels_ang_vel` but acts on the ``yaw_moment`` and ``roll_moment``
    ranges of :class:`UniformMomentCommandCfg`. Starting from ``range_multiplier[0]`` of the full
    range, the ranges expand toward ``range_multiplier[1]`` in ``num_steps`` successful episodes,
    so the policy learns to drive/turn before it is asked to bank hard.
    """
    ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    if env.common_step_counter == 0:
        env._orig_yaw_moment = torch.tensor(ranges.yaw_moment, device=env.device)
        env._orig_roll_moment = torch.tensor(ranges.roll_moment, device=env.device)
        env._init_yaw_moment = env._orig_yaw_moment * range_multiplier[0]
        env._final_yaw_moment = env._orig_yaw_moment * range_multiplier[1]
        env._init_roll_moment = env._orig_roll_moment * range_multiplier[0]
        env._final_roll_moment = env._orig_roll_moment * range_multiplier[1]
        env._delta_yaw_moment = (env._final_yaw_moment - env._init_yaw_moment) / num_steps
        env._delta_roll_moment = (env._final_roll_moment - env._init_roll_moment) / num_steps
        ranges.yaw_moment = env._init_yaw_moment.tolist()
        ranges.roll_moment = env._init_roll_moment.tolist()

    # update only on episode boundaries since the max command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.5 * reward_term_cfg.weight:
            new_yaw = torch.tensor(ranges.yaw_moment, device=env.device) + env._delta_yaw_moment
            new_roll = torch.tensor(ranges.roll_moment, device=env.device) + env._delta_roll_moment
            new_yaw = torch.clamp(new_yaw, min=env._final_yaw_moment[0], max=env._final_yaw_moment[1])
            new_roll = torch.clamp(new_roll, min=env._final_roll_moment[0], max=env._final_roll_moment[1])
            ranges.yaw_moment = new_yaw.tolist()
            ranges.roll_moment = new_roll.tolist()

    return torch.tensor(ranges.roll_moment[1], device=env.device)


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_lin_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_ang_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original angular velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)

def command_levels_lin_vel_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    initial_vel_x: tuple[float, float] = (-0.4, 0.4),
    final_vel_x: tuple[float, float] = (-1.5, 4.0),
    initial_vel_y: tuple[float, float] = (-0.02, 0.02),
    final_vel_y: tuple[float, float] = (-0.2, 0.2),
    num_steps: int = 10,
) -> torch.Tensor:
    """Curriculum that gradually expands linear velocity command ranges.

    Unlike :func:`command_levels_lin_vel` which uses a symmetric multiplier,
    this function accepts explicit initial and final ranges for each velocity
    component, allowing asymmetric expansion.

    On each episode boundary, if the average tracking reward exceeds 80% of
    the reward term weight, the command range is expanded by one step toward
    the final range. The total expansion takes ``num_steps`` successful
    episodes.
    """
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    if env.common_step_counter == 0:
        env._curr_initial_vel_x = torch.tensor(initial_vel_x, device=env.device)
        env._curr_final_vel_x = torch.tensor(final_vel_x, device=env.device)
        env._curr_initial_vel_y = torch.tensor(initial_vel_y, device=env.device)
        env._curr_final_vel_y = torch.tensor(final_vel_y, device=env.device)
        env._curr_delta_vel_x = (env._curr_final_vel_x - env._curr_initial_vel_x) / num_steps
        env._curr_delta_vel_y = (env._curr_final_vel_y - env._curr_initial_vel_y) / num_steps

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = list(initial_vel_x)
        base_velocity_ranges.lin_vel_y = list(initial_vel_y)

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + env._curr_delta_vel_x
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + env._curr_delta_vel_y

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._curr_final_vel_x[0], max=env._curr_final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._curr_final_vel_y[0], max=env._curr_final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    initial_ang_vel_z: tuple[float, float] = (-0.4, 0.4),
    final_ang_vel_z: tuple[float, float] = (-3.14159, 3.14159),
    num_steps: int = 10,
) -> torch.Tensor:
    """Curriculum that gradually expands angular velocity command ranges.

    Unlike :func:`command_levels_ang_vel` which uses a symmetric multiplier,
    this function accepts explicit initial and final ranges, allowing
    asymmetric expansion.

    On each episode boundary, if the average tracking reward exceeds 80% of
    the reward term weight, the command range is expanded by one step toward
    the final range. The total expansion takes ``num_steps`` successful
    episodes.
    """
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    if env.common_step_counter == 0:
        env._curr_initial_ang_vel_z = torch.tensor(initial_ang_vel_z, device=env.device)
        env._curr_final_ang_vel_z = torch.tensor(final_ang_vel_z, device=env.device)
        env._curr_delta_ang_vel_z = (env._curr_final_ang_vel_z - env._curr_initial_ang_vel_z) / num_steps

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = list(initial_ang_vel_z)

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.4 * reward_term_cfg.weight: # be tolerant on angular velocity tracking since it's generally harder to learn
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + env._curr_delta_ang_vel_z

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.clamp(
                new_ang_vel_z, min=env._curr_final_ang_vel_z[0], max=env._curr_final_ang_vel_z[1]
            )

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)
