from __future__ import annotations
import numpy as np
import torch

LEG_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
WHEEL_JOINTS = ["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"]
JOINT_ORDER = LEG_JOINTS + WHEEL_JOINTS

DEFAULT_Q = np.array([0.0, 0.8, -1.5] * 4 + [0.0, 0.0, 0.0, 0.0], dtype=np.float64)
LEG_SCALE = np.array([0.125 if "hip" in j else 0.25 for j in LEG_JOINTS], dtype=np.float64)
WHEEL_VEL_SCALE = 5.0
LIN_VEL_SCALE = 2.0
ANG_VEL_SCALE = 0.25
GRAVITY_SCALE = 1.0
JOINT_POS_SCALE = 1.0
JOINT_VEL_SCALE = 0.05
SETPOINT_SCALE = np.array([2.0, 0.25, 1.0])

OBS_DIM = 56
ACTION_DIM = 16
N_LEGS = 12

M = 15.0
H = 0.4
IX = 2.5
IZ = 2.8
GRAV = 9.81
V_MIN = 0.0
V_MAX = 3.0
WZ_MAX = float(np.pi / 3)
THETA_MAX = 0.2
CMD_DEADBAND = 0.05
IDLE_DECAY_TAU = 0.4


class SetpointIntegrator:
    def __init__(self, dt):
        self.dt = dt
        self.reset()

    def reset(self):
        self.v = 0.0
        self.wz = 0.0
        self.theta = 0.0
        self.theta_dot = 0.0

    def update(self, command):
        a_x, m_psi, m_theta = float(command[0]), float(command[1]), float(command[2])
        dt = self.dt
        v_int = np.clip(self.v + a_x * dt, V_MIN, V_MAX)
        wz_int = np.clip(self.wz + (m_psi / IZ) * dt, -WZ_MAX, WZ_MAX)
        a_y = self.v * self.wz
        th_dd = (-M * GRAV * H * np.sin(self.theta) - M * a_y * H * np.cos(self.theta) + m_theta) / IX
        thdot_int = self.theta_dot + th_dd * dt
        th_int = float(np.clip(self.theta + thdot_int * dt, -THETA_MAX, THETA_MAX))

        if abs(a_x) < CMD_DEADBAND and abs(m_psi) < CMD_DEADBAND and abs(m_theta) < CMD_DEADBAND:
            decay = np.exp(-dt / IDLE_DECAY_TAU)
            self.v *= decay
            self.wz *= decay
            self.theta *= decay
            self.theta_dot *= decay
        else:
            self.v = v_int
            self.wz = wz_int
            self.theta = th_int
            self.theta_dot = thdot_int
        return np.array([self.v, self.wz, self.theta])


class PolicyRunner:
    def __init__(self, policy_path, device="cpu", dt=0.02, action_ema=0.4):
        self.device = device
        self.policy = torch.jit.load(policy_path, map_location=device).eval()
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.integ = SetpointIntegrator(dt)
        self.action_ema = action_ema
        self._af = None

    def reset(self):
        self.last_action[:] = 0.0
        self.integ.reset()
        self._af = None

    def build_obs(self, base_lin_vel, base_ang_vel, projected_gravity, command, joint_pos, joint_vel):
        setpoints = self.integ.update(command) * SETPOINT_SCALE
        leg_pos_rel = np.asarray(joint_pos, dtype=np.float64)[:N_LEGS] - DEFAULT_Q[:N_LEGS]
        obs = np.concatenate([
            np.asarray(base_lin_vel, dtype=np.float64) * LIN_VEL_SCALE,
            np.asarray(base_ang_vel, dtype=np.float64) * ANG_VEL_SCALE,
            np.asarray(projected_gravity, dtype=np.float64) * GRAVITY_SCALE,
            setpoints,
            leg_pos_rel * JOINT_POS_SCALE,
            np.asarray(joint_vel, dtype=np.float64) * JOINT_VEL_SCALE,
            self.last_action,
        ])
        assert obs.shape[0] == OBS_DIM, f"obs dim {obs.shape[0]} != {OBS_DIM}"
        return obs.astype(np.float32)

    @torch.no_grad()
    def act(self, obs):
        t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(self.device)
        a = self.policy(t).squeeze(0).cpu().numpy().astype(np.float64)
        # low-pass the action to smooth the jittery wheel-velocity targets
        self._af = a.copy() if self._af is None else self.action_ema * a + (1 - self.action_ema) * self._af
        a = self._af
        self.last_action = a.copy()
        return a

    def decode(self, action):
        a = np.asarray(action, dtype=np.float64)
        leg_q_des = DEFAULT_Q[:N_LEGS] + LEG_SCALE * a[:N_LEGS]
        wheel_dq_des = WHEEL_VEL_SCALE * a[N_LEGS:]
        return leg_q_des, wheel_dq_des
