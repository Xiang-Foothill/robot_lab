# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects "pits" terrain and applies restrictions:
    - For pit terrains: only allow forward movement (no lateral or rotational movement)
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # Track which robots were on pit terrain in the previous step
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands with threshold."""
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        # First, call parent's update command
        super()._update_command()

        # Check which robots are currently on pit terrain (real-time check every step)
        on_pits = is_robot_on_terrain(self._env, "pits")

        # Find robots that just left pit terrain (need to resample)
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            left_pit_env_ids = torch.where(left_pit_mask)[0]
            # Resample commands for robots that left pits
            self._resample_command(left_pit_env_ids)

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            # Force forward-only movement with min and max speed limits
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement
            self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation
            # Set heading to 0 for pit robots
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0

        # Update tracking state
        self.was_on_pit = on_pits


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """


class UniformDifferentialCommand(CommandTerm):
    """Command generator for differential-drive control: [vx, roll_rate, wz].

    vx        — longitudinal velocity (m/s)
    roll_rate — body roll angular velocity (rad/s), typically commanded to 0
    wz        — yaw rate (rad/s)
    """

    cfg: "UniformDifferentialCommandCfg"

    def __init__(self, cfg: "UniformDifferentialCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._cmd = torch.zeros(self.num_envs, 3, device=self.device)

    def __str__(self) -> str:
        return (
            f"UniformDifferentialCommand:\n"
            f"\tvx:        {self.cfg.ranges.lin_vel_x}\n"
            f"\troll_rate: {self.cfg.ranges.roll_rate}\n"
            f"\twz:        {self.cfg.ranges.ang_vel_z}\n"
        )

    @property
    def command(self) -> torch.Tensor:
        return self._cmd

    def _resample_command(self, env_ids: Sequence[int]):
        r = self.cfg.ranges
        n = len(env_ids)
        self._cmd[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*r.lin_vel_x)
        self._cmd[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*r.roll_rate)
        self._cmd[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*r.ang_vel_z)
        self._cmd[env_ids, 0] *= (torch.abs(self._cmd[env_ids, 0]) > 0.2).float()
        self._cmd[env_ids, 2] *= (torch.abs(self._cmd[env_ids, 2]) > 0.1).float()
        if self.cfg.rel_standing_envs > 0.0:
            standing = torch.rand(n, device=self.device) < self.cfg.rel_standing_envs
            env_ids_t = torch.as_tensor(env_ids, device=self.device)
            self._cmd[env_ids_t[standing], :] = 0.0

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass


@configclass
class UniformDifferentialCommandCfg(CommandTermCfg):
    """Configuration for UniformDifferentialCommand."""

    class_type: type = UniformDifferentialCommand

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = (-2.0, 2.0)
        roll_rate: tuple[float, float] = (0.0, 0.0)
        ang_vel_z: tuple[float, float] = (-1.0, 1.0)

    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    ranges: Ranges = Ranges()
    rel_standing_envs: float = 0.0


class UniformMomentCommand(CommandTerm):
    """Command generator for the paper's MPC command ``[a_x, M_psi, M_theta]``.

    a_x      — longitudinal acceleration command (m/s^2)
    M_psi    — yaw moment command (N*m)
    M_theta  — roll moment command (N*m)

    The raw 3-vector is exposed verbatim by :attr:`command` and is what the policy observes,
    matching the paper. Internally the term integrates these commands through the paper's own
    rigid-body model (Sec. 3) into reference body states — target speed ``v*``, target yaw rate
    ``wz*`` and target roll angle ``theta*`` — which the setpoint-tracking reward functions read.
    Because ``theta*`` is the *steady-state* lean produced by a sustained ``M_theta`` (the roll
    ODE, Eq. roll), holding a bank yields a nonzero reward gradient, which is what lets the policy
    learn to tilt (unlike commanding a roll *rate*, which is zero at a held lean).
    """

    cfg: "UniformMomentCommandCfg"

    def __init__(self, cfg: "UniformMomentCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        # raw command [a_x, M_psi, M_theta]
        self._cmd = torch.zeros(self.num_envs, 3, device=self.device)
        # integrated reference states
        self._v_star = torch.zeros(self.num_envs, device=self.device)
        self._wz_star = torch.zeros(self.num_envs, device=self.device)
        self._theta_star = torch.zeros(self.num_envs, device=self.device)
        self._theta_dot = torch.zeros(self.num_envs, device=self.device)
        # paper constants
        p = cfg.params
        self._m, self._h = p.mass, p.cg_height
        self._Ix, self._Iz, self._g = p.roll_inertia, p.yaw_inertia, p.gravity

    def __str__(self) -> str:
        return (
            f"UniformMomentCommand:\n"
            f"\ta_x:     {self.cfg.ranges.lin_accel_x}\n"
            f"\tM_psi:   {self.cfg.ranges.yaw_moment}\n"
            f"\tM_theta: {self.cfg.ranges.roll_moment}\n"
        )

    @property
    def command(self) -> torch.Tensor:
        """Raw MPC command ``[a_x, M_psi, M_theta]`` exposed to the observation. Shape (N, 3)."""
        return self._cmd

    # -- reference-state accessors for the reward functions --
    @property
    def v_star(self) -> torch.Tensor:
        return self._v_star

    @property
    def wz_star(self) -> torch.Tensor:
        return self._wz_star

    @property
    def theta_star(self) -> torch.Tensor:
        return self._theta_star

    def _measured_roll(self, env_ids_t: torch.Tensor) -> torch.Tensor:
        quat = wp.to_torch(self.robot.data.root_quat_w)[env_ids_t]
        roll, _, _ = math_utils.euler_xyz_from_quat(quat)
        return math_utils.wrap_to_pi(roll)

    def _resample_command(self, env_ids: Sequence[int]):
        r = self.cfg.ranges
        n = len(env_ids)
        dev = self.device
        env_ids_t = torch.as_tensor(env_ids, device=dev)

        a_x = torch.empty(n, device=dev).uniform_(*r.lin_accel_x)
        # small longitudinal-accel deadband
        a_x = a_x * (a_x.abs() > 0.1).float()
        m_psi = torch.empty(n, device=dev).uniform_(*r.yaw_moment)

        # seed the reference states from the measured body state to avoid large startup error
        v_meas = wp.to_torch(self.robot.data.root_lin_vel_b)[env_ids_t, 0]
        wz_meas = wp.to_torch(self.robot.data.root_ang_vel_b)[env_ids_t, 2]
        roll_meas = self._measured_roll(env_ids_t)
        self._v_star[env_ids_t] = v_meas
        self._wz_star[env_ids_t] = wz_meas
        self._theta_star[env_ids_t] = roll_meas
        self._theta_dot[env_ids_t] = 0.0

        # Tier 3: draw M_theta near the LTR-cancelling roll moment implied by the commanded yaw.
        # The anticipated yaw rate after a short lookahead under M_psi:
        wz_imp = torch.clamp(
            wz_meas + (m_psi / self._Iz) * self.cfg.corr_lookahead, -self.cfg.wz_max, self.cfg.wz_max
        )
        # steady-state roll moment that cancels lateral load transfer: M ~= -m*h*v*psi_dot
        m_theta0 = -self._m * self._h * v_meas * wz_imp
        noise = torch.empty(n, device=dev).uniform_(*r.roll_moment_noise)
        m_theta = torch.clamp(m_theta0 + noise, r.roll_moment[0], r.roll_moment[1])
        # keep a fraction fully uniform for coverage off the raceline manifold
        rand_mask = torch.rand(n, device=dev) < self.cfg.p_random_roll
        m_theta_rand = torch.empty(n, device=dev).uniform_(*r.roll_moment)
        m_theta = torch.where(rand_mask, m_theta_rand, m_theta)

        self._cmd[env_ids_t, 0] = a_x
        self._cmd[env_ids_t, 1] = m_psi
        self._cmd[env_ids_t, 2] = m_theta

        if self.cfg.rel_standing_envs > 0.0:
            standing = torch.rand(n, device=dev) < self.cfg.rel_standing_envs
            self._cmd[env_ids_t[standing], :] = 0.0

    def _update_command(self):
        # integrate the raw commands into reference states using the paper's rigid-body model
        dt = self._env.step_dt
        a_x = self._cmd[:, 0]
        m_psi = self._cmd[:, 1]
        m_theta = self._cmd[:, 2]

        self._v_star = torch.clamp(self._v_star + a_x * dt, self.cfg.v_min, self.cfg.v_max)
        self._wz_star = torch.clamp(self._wz_star + (m_psi / self._Iz) * dt, -self.cfg.wz_max, self.cfg.wz_max)

        # roll dynamics: I_x*theta_ddot = -m*g*h*sin(theta) - m*a_y*h*cos(theta) + M_theta
        a_y = self._v_star * self._wz_star
        th = self._theta_star
        th_dd = (
            -self._m * self._g * self._h * torch.sin(th)
            - self._m * a_y * self._h * torch.cos(th)
            + m_theta
        ) / self._Ix
        self._theta_dot = self._theta_dot + th_dd * dt
        self._theta_star = torch.clamp(th + self._theta_dot * dt, -self.cfg.theta_max, self.cfg.theta_max)

    def _update_metrics(self):
        pass


@configclass
class UniformMomentCommandCfg(CommandTermCfg):
    """Configuration for :class:`UniformMomentCommand`."""

    class_type: type = UniformMomentCommand
    asset_name: str = "robot"

    @configclass
    class Ranges:
        lin_accel_x: tuple[float, float] = (-2.0, 2.0)  # a_x (m/s^2)
        yaw_moment: tuple[float, float] = (-8.0, 8.0)  # M_psi (N*m)
        roll_moment: tuple[float, float] = (-15.0, 15.0)  # M_theta (N*m), |M_theta| <= M_theta,max
        roll_moment_noise: tuple[float, float] = (-5.0, 5.0)  # noise around the LTR-cancelling center

    @configclass
    class Params:
        mass: float = 15.0  # m (kg)
        cg_height: float = 0.40  # h (m)
        roll_inertia: float = 2.5  # I_x (kg*m^2)
        yaw_inertia: float = 2.8  # I_z (kg*m^2)
        gravity: float = 9.81  # g (m/s^2)

    ranges: Ranges = Ranges()
    params: Params = Params()
    resampling_time_range: tuple[float, float] = (1.5, 4.0)
    rel_standing_envs: float = 0.05
    p_random_roll: float = 0.25  # fraction of fully-uniform M_theta samples
    corr_lookahead: float = 0.3  # s, yaw-rate lookahead for LTR-correlated M_theta
    v_min: float = 0.0
    v_max: float = 3.0
    wz_max: float = math.pi / 3
    theta_max: float = 0.5
