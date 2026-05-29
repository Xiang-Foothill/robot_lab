# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


OBS_DELAY_JOINT_POS_KEY = "joint_pos"
OBS_DELAY_JOINT_VEL_KEY = "joint_vel"
OBS_DELAY_BASE_ANG_VEL_KEY = "base_ang_vel"
OBS_DELAY_KEYS = (OBS_DELAY_JOINT_POS_KEY, OBS_DELAY_JOINT_VEL_KEY, OBS_DELAY_BASE_ANG_VEL_KEY)


def _ensure_observation_delay_state(env: ManagerBasedEnv):
    if not hasattr(env, "_obs_delay_max_steps"):
        env._obs_delay_max_steps = 0
    if not hasattr(env, "_obs_delay_enabled"):
        env._obs_delay_enabled = False
    if not hasattr(env, "_obs_delay_buffers"):
        env._obs_delay_buffers = {}
    if not hasattr(env, "_obs_delay_buffer_size"):
        env._obs_delay_buffer_size = 1
    if not hasattr(env, "_obs_delay_head"):
        env._obs_delay_head = 0
    if not hasattr(env, "_obs_delay_last_policy_step"):
        env._obs_delay_last_policy_step = -1

    for key in OBS_DELAY_KEYS:
        attr_name = f"_obs_delay_{key}_steps"
        if not hasattr(env, attr_name):
            setattr(env, attr_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.long))


def _ensure_observation_delay_buffers(env: ManagerBasedEnv):
    _ensure_observation_delay_state(env)
    buffer_size = int(env._obs_delay_max_steps) + 1
    if buffer_size < 1:
        buffer_size = 1

    asset: Articulation = env.scene["robot"]
    joint_dim = wp.to_torch(asset.data.joint_pos).shape[1]

    expected_shapes = {
        OBS_DELAY_JOINT_POS_KEY: (env.num_envs, buffer_size, joint_dim),
        OBS_DELAY_JOINT_VEL_KEY: (env.num_envs, buffer_size, joint_dim),
        OBS_DELAY_BASE_ANG_VEL_KEY: (env.num_envs, buffer_size, 3),
    }

    if env._obs_delay_buffer_size != buffer_size:
        env._obs_delay_buffers = {}
        env._obs_delay_head = 0

    for key, shape in expected_shapes.items():
        needs_init = (key not in env._obs_delay_buffers) or (env._obs_delay_buffers[key].shape != shape)
        if needs_init:
            env._obs_delay_buffers[key] = torch.zeros(shape, device=env.device, dtype=torch.float)

    env._obs_delay_buffer_size = buffer_size


def reset_observation_delay_buffers(env: ManagerBasedEnv, env_ids: torch.Tensor | None = None):
    _ensure_observation_delay_state(env)
    _ensure_observation_delay_buffers(env)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    for key in OBS_DELAY_KEYS:
        env._obs_delay_buffers[key][env_ids] = 0.0


def record_observation_delay_sample(env: ManagerBasedEnv, force: bool = False):
    """Record one sample for delay buffers.

    If ``force=False``, this records at most once per policy step, which is the safe fallback when
    physics-step hooks are unavailable. If ``force=True``, recording happens on every call and can
    be used from physics-step callbacks.
    """

    _ensure_observation_delay_state(env)
    if not env._obs_delay_enabled:
        return

    _ensure_observation_delay_buffers(env)

    if not force:
        policy_step = int(getattr(env, "common_step_counter", -1))
        if env._obs_delay_last_policy_step == policy_step:
            return
        env._obs_delay_last_policy_step = policy_step

    asset: Articulation = env.scene["robot"]
    joint_pos_rel = wp.to_torch(asset.data.joint_pos) - wp.to_torch(asset.data.default_joint_pos)
    joint_vel = wp.to_torch(asset.data.joint_vel)
    base_ang_vel = wp.to_torch(asset.data.root_ang_vel_b)

    env._obs_delay_head = (env._obs_delay_head + 1) % env._obs_delay_buffer_size
    head = env._obs_delay_head

    env._obs_delay_buffers[OBS_DELAY_JOINT_POS_KEY][:, head, :] = joint_pos_rel
    env._obs_delay_buffers[OBS_DELAY_JOINT_VEL_KEY][:, head, :] = joint_vel
    env._obs_delay_buffers[OBS_DELAY_BASE_ANG_VEL_KEY][:, head, :] = base_ang_vel


def _get_delayed_from_buffer(env: ManagerBasedEnv, key: str) -> torch.Tensor:
    _ensure_observation_delay_state(env)
    if not env._obs_delay_enabled:
        asset: Articulation = env.scene["robot"]
        if key == OBS_DELAY_JOINT_POS_KEY:
            return wp.to_torch(asset.data.joint_pos) - wp.to_torch(asset.data.default_joint_pos)
        if key == OBS_DELAY_JOINT_VEL_KEY:
            return wp.to_torch(asset.data.joint_vel)
        if key == OBS_DELAY_BASE_ANG_VEL_KEY:
            return wp.to_torch(asset.data.root_ang_vel_b)
        raise RuntimeError(f"Unknown observation delay key: {key}")

    record_observation_delay_sample(env, force=False)

    delays = getattr(env, f"_obs_delay_{key}_steps")
    delays = torch.clamp(delays, min=0, max=env._obs_delay_max_steps)
    idx = (env._obs_delay_head - delays) % env._obs_delay_buffer_size
    env_ids = torch.arange(env.num_envs, device=env.device)
    return env._obs_delay_buffers[key][env_ids, idx, :]


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = wp.to_torch(asset.data.joint_pos)[:, asset_cfg.joint_ids] - wp.to_torch(asset.data.default_joint_pos)[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def joint_pos_rel_delayed(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    delayed_joint_pos_rel = _get_delayed_from_buffer(env, OBS_DELAY_JOINT_POS_KEY)
    return delayed_joint_pos_rel[:, asset_cfg.joint_ids]


def joint_pos_rel_without_wheel_delayed(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    delayed_joint_pos_rel = _get_delayed_from_buffer(env, OBS_DELAY_JOINT_POS_KEY)[:, asset_cfg.joint_ids]
    delayed_joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return delayed_joint_pos_rel


def joint_vel_rel_delayed(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    delayed_joint_vel = _get_delayed_from_buffer(env, OBS_DELAY_JOINT_VEL_KEY)
    return delayed_joint_vel[:, asset_cfg.joint_ids]


def base_ang_vel_delayed(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    del asset_cfg
    return _get_delayed_from_buffer(env, OBS_DELAY_BASE_ANG_VEL_KEY)
