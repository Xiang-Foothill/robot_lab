# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""Standalone MuJoCo test: drive the Go2W smooth-steer policy around the raceline."""

from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import numpy as np

from accel_pursuit_planner import AccelPursuitPlanner
from policy_runner import DEFAULT_Q, JOINT_ORDER, PolicyRunner

_PLANNER_NODES = os.path.expanduser(
    "~/go2w_planner_ws/src/planner_nodes/planner_nodes")
if _PLANNER_NODES not in sys.path:
    sys.path.insert(0, _PLANNER_NODES)

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "assets", "go2w.xml")
DEFAULT_RACELINE = os.path.expanduser("~/go2w_planner_ws/data/raceline_ltrack.npz")
DEFAULT_POLICY = os.path.expanduser(
    "~/Repositories/robot_lab/rsl_rl/unitree_go2w_smooth_steer/2026-05-29_08-49-22/exported/policy.pt"
)

PHYS_DT = 0.005
CONTROL_DECIMATION = 4
PLAN_EVERY_N_CONTROL = 5


def quat_to_yaw(q):
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class Sim:
    def __init__(self, xml, raceline, policy_path, planner="pursuit", v_max=1.5):
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        if planner == "mpc":
            from fw_mpc_core import BicycleMPCAdapter
            self.planner = BicycleMPCAdapter(raceline, v_max=v_max)
        else:
            self.planner = AccelPursuitPlanner(raceline)
        self.runner = PolicyRunner(policy_path)

        self.qpos_adr = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in JOINT_ORDER])
        self.qvel_adr = np.array([
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in JOINT_ORDER])
        self.act_idx = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{j}")
            for j in JOINT_ORDER])

        self.sens = {}
        for name in ("imu_gyro", "imu_vel", "base_quat", "base_pos"):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self.sens[name] = (self.model.sensor_adr[sid], self.model.sensor_dim[sid])

        d = np.load(raceline)
        self.race_pts = d["pts"].astype(float)
        self.race_theta = d["theta"].astype(float)
        self.bound_inner = d["bound_inner"].astype(float) if "bound_inner" in d.files else None
        self.bound_outer = d["bound_outer"].astype(float) if "bound_outer" in d.files else None

    def _sensor(self, name):
        adr, dim = self.sens[name]
        return self.data.sensordata[adr:adr + dim].copy()

    def reset_on_track(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_adr] = DEFAULT_Q
        x0, y0 = self.race_pts[0]
        yaw0 = float(self.race_theta[0])
        self.data.qpos[0:3] = [x0, y0, 0.45]
        self.data.qpos[3:7] = yaw_to_quat(yaw0)
        mujoco.mj_forward(self.model, self.data)
        self.runner.reset()

    def read_planner_state(self):
        pos = self._sensor("base_pos")
        quat = self._sensor("base_quat")
        vbody = self._sensor("imu_vel")
        return float(pos[0]), float(pos[1]), quat_to_yaw(quat), float(vbody[0])

    def read_policy_state(self):
        lin_vel = self._sensor("imu_vel")
        gyro = self._sensor("imu_gyro")
        jpos = self.data.qpos[self.qpos_adr].copy()
        jvel = self.data.qvel[self.qvel_adr].copy()
        return lin_vel, gyro, jpos, jvel

    def apply_action(self, action):
        leg_q_des, wheel_dq_des = self.runner.decode(action)
        self.data.ctrl[self.act_idx] = np.concatenate([leg_q_des, wheel_dq_des])

    def run(self, duration, viewer=False, verbose=True, real_time=True):
        n_control = int(duration / (PHYS_DT * CONTROL_DECIMATION))
        control_dt = PHYS_DT * CONTROL_DECIMATION
        log = {"t": [], "x": [], "y": [], "v": [], "v_ref": [], "cmd": [], "z": [], "s": []}
        command = np.zeros(3)
        last_info = {"v_ref": 0.0, "s": 0.0}

        vh = None
        if viewer:
            from mujoco import viewer as mj_viewer
            vh = mj_viewer.launch_passive(self.model, self.data)
            base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
            if base_id >= 0:
                vh.cam.trackbodyid = base_id
                vh.cam.distance = 4.0
                vh.cam.azimuth = 90.0
                vh.cam.elevation = -25.0

        self.reset_on_track()
        wall_start = time.perf_counter()
        try:
            for k in range(n_control):
                x, y, psi, v = self.read_planner_state()
                if k % PLAN_EVERY_N_CONTROL == 0:
                    command, last_info = self.planner.plan(x, y, psi, v)

                lin_vel, gyro, jpos, jvel = self.read_policy_state()
                obs = self.runner.build_obs(lin_vel, gyro, command, jpos, jvel)
                action = self.runner.act(obs)
                self.apply_action(action)
                for _ in range(CONTROL_DECIMATION):
                    mujoco.mj_step(self.model, self.data)

                t = k * PHYS_DT * CONTROL_DECIMATION
                log["t"].append(t); log["x"].append(x); log["y"].append(y)
                log["v"].append(v); log["v_ref"].append(last_info["v_ref"])
                log["cmd"].append(command.copy()); log["z"].append(float(self.data.qpos[2]))
                log["s"].append(last_info["s"])

                if vh is not None:
                    vh.sync()
                if real_time and vh is not None:
                    lag = (k + 1) * control_dt - (time.perf_counter() - wall_start)
                    if lag > 0:
                        time.sleep(lag)
                if verbose and k % 50 == 0:
                    print(f"t={t:5.1f}s  s={last_info['s']:5.2f}/{self.planner.track_length:.1f}m  "
                          f"pos=({x:5.2f},{y:5.2f})  v={v:4.2f}->{last_info['v_ref']:4.2f}  "
                          f"cmd[vx,roll,wz]=[{command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f}]  "
                          f"z={self.data.qpos[2]:.3f}")
                if self.data.qpos[2] < 0.15:
                    print(f"!! base height {self.data.qpos[2]:.3f} m - robot fell at t={t:.1f}s")
                    break
        finally:
            if vh is not None:
                vh.close()
        return {k: np.array(v) for k, v in log.items()}

    def report(self, log, save_path):
        from scipy.spatial import KDTree
        xy = np.stack([log["x"], log["y"]], axis=1)
        tree = KDTree(self.race_pts)
        dists, _ = tree.query(xy)
        s = log["s"]
        progress = np.max(s) if len(s) else 0.0
        wraps = np.sum(np.diff(s) < -self.planner.track_length / 2)
        print("\n=== RESULT ===")
        print(f"  sim time:            {log['t'][-1]:.1f} s")
        print(f"  arclength reached:   {progress:.2f} m (track {self.planner.track_length:.2f} m), lap wraps: {int(wraps)}")
        print(f"  cross-track error:   mean {dists.mean():.3f} m  max {dists.max():.3f} m")
        print(f"  speed:               mean {log['v'].mean():.2f}  max {log['v'].max():.2f} m/s")
        print(f"  base height:         mean {log['z'].mean():.3f}  min {log['z'].min():.3f} m")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(16, 5))
            if self.bound_inner is not None:
                ax[0].plot(self.bound_inner[:, 0], self.bound_inner[:, 1], "k", lw=1)
                ax[0].plot(self.bound_outer[:, 0], self.bound_outer[:, 1], "k", lw=1, label="track")
            ax[0].plot(self.race_pts[:, 0], self.race_pts[:, 1], "g--", lw=1, label="raceline")
            ax[0].plot(log["x"], log["y"], "b-", lw=1.5, label="robot")
            ax[0].plot(log["x"][0], log["y"][0], "go", label="start")
            ax[0].axis("equal"); ax[0].legend(); ax[0].set_title("Trajectory")
            ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
            ax[1].plot(log["t"], log["v"], label="v measured")
            ax[1].plot(log["t"], log["v_ref"], "--", label="v ref")
            ax[1].plot(log["t"], log["cmd"][:, 0], ":", label="vx cmd")
            ax[1].legend(); ax[1].set_title("Speed / vel cmd"); ax[1].set_xlabel("t [s]")
            ax[2].plot(log["t"], dists, label="cross-track err")
            ax[2].plot(log["t"], log["cmd"][:, 2], "--", label="wz cmd")
            ax[2].legend(); ax[2].set_title("Tracking error / yaw cmd"); ax[2].set_xlabel("t [s]")
            fig.tight_layout(); fig.savefig(save_path, dpi=110)
            print(f"  saved plot -> {save_path}")
        except Exception as e:
            print(f"  (plot skipped: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=XML)
    ap.add_argument("--raceline", default=DEFAULT_RACELINE)
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--planner", choices=["pursuit", "mpc"], default="pursuit")
    ap.add_argument("--v_max", type=float, default=1.5)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--no_real_time", action="store_true")
    ap.add_argument("--plot", default=os.path.join(HERE, "track_result.png"))
    args = ap.parse_args()

    sim = Sim(args.xml, args.raceline, args.policy,
              planner=args.planner, v_max=args.v_max)
    log = sim.run(args.duration, viewer=args.viewer, real_time=not args.no_real_time)
    sim.report(log, args.plot)


if __name__ == "__main__":
    main()
