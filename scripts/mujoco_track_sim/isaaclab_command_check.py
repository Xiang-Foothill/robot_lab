# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""Force a fixed command in the native IsaacLab smooth-steer env and measure the achieved
base rates, to ground-truth a policy before MuJoCo/robot deployment."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ax", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=1.0)
parser.add_argument("--steps", type=int, default=250)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.num_envs = 1
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import warp as wp

import robot_lab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

TASK = "RobotLab-Isaac-Velocity-SmoothSteer-Unitree-Go2W-v0"
POLICY = "/home/marla/robot_lab/logs/rsl_rl/unitree_go2w_smooth_steer/2026-05-13_16-50-51/exported/policy.pt"

env_cfg = parse_env_cfg(TASK, num_envs=1)
env_cfg.commands.base_velocity.ranges.lin_acc_x = (args.ax, args.ax)
env_cfg.commands.base_velocity.ranges.roll_rate = (0.0, 0.0)
env_cfg.commands.base_velocity.ranges.ang_vel_z = (args.wz, args.wz)

env = gym.make(TASK, cfg=env_cfg)
obs, _ = env.reset()
policy = torch.jit.load(POLICY, map_location=env.unwrapped.device).eval()
robot = env.unwrapped.scene["robot"]

obs_t = obs["policy"]
yaw_rates, ax_body = [], []
for i in range(args.steps):
    with torch.no_grad():
        action = policy(obs_t)
    obs, _, _, _, _ = env.step(action)
    obs_t = obs["policy"]
    yaw_rates.append(float(wp.to_torch(robot.data.root_ang_vel_b)[0, 2]))
    if i == 0:
        cmd = env.unwrapped.command_manager.get_command("base_velocity")[0].cpu().numpy()
        print(f"[isaaclab] forced command = {cmd}  obs dim {tuple(obs_t.shape)}", flush=True)
        print(f"[isaaclab] obs[:6] = {obs_t[0, :6].cpu().numpy()}", flush=True)

yr = np.array(yaw_rates[-100:])
print(f"\n=== ISAACLAB RESULT (ax={args.ax} wz={args.wz}) ===", flush=True)
print(f"  achieved yaw rate: mean {yr.mean():+.3f}  std {yr.std():.3f} rad/s", flush=True)
print(f"  -> policy {'DOES' if abs(yr.mean()) > 0.3 else 'does NOT'} produce strong yaw", flush=True)

env.close()
simulation_app.close()
