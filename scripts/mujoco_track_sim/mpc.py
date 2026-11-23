import os
import numpy as np
import casadi as ca
from scipy.spatial import KDTree

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RACELINE = os.path.join(HERE, "assets", "raceline.h5")

HORIZON = 35
DT_MPC = 0.1
N_SUBSTEP = 10
V_MAX = 2.0
V_MIN = 0.3
AX_MAX = 0.8
WZ_MAX = float(np.pi / 3)
V_TURN = 0.2
G = 9.81

W_LATERAL = 2000.0
W_HEADING = 300.0
W_SPEED = 40.0
W_BETA = 200.0
W_YAW = 2000.0
W_BOUNDS = 16000.0
W_LTR = 50.0
W_ROLL_BOUND = 5000.0
BETA_MAX = 0.3
ROLL_MAX = 0.3
CTE_SOFT_M = 0.6
M_ROLL_MAX = 15.0
R_CTRL = np.diag([0.01, 0.001, 0.001])
R_RATE = np.diag([10.0, 1.0, 1.0])

_ROBOT_PARAMS = {'Caf': 800.0, 'Car': 750.0, 'm': 15.0, 'Lf': 0.2, 'Lr': 0.3,
                 'Iz': 2.8, 'Ix': 2.5, 'h': 0.4, 'g': 9.81}


def _load_track(path):
    import h5py
    with h5py.File(path, 'r') as h5f:
        x = np.asarray(h5f['x'], dtype=float)
        y = np.asarray(h5f['y'], dtype=float)
        theta = np.asarray(h5f['psi'], dtype=float)
        v = np.asarray(h5f['v'], dtype=float)
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    ss = np.cumsum(np.hypot(dx, dy))
    return {
        'x': x, 'y': y, 'theta': theta,
        'nx': -np.sin(theta), 'ny': np.cos(theta),
        'v': v, 's': ss, 'L': float(ss[-1]),
    }


class _BicycleMPC:
    # state: x y psi beta dpsi v theta dtheta
    # control: a_x, yaw moment, roll moment

    def __init__(self, track, v_max=V_MAX, v_min=V_MIN, params=_ROBOT_PARAMS):
        self.track = track
        self.v_max = v_max
        self.v_min = v_min
        self.p = params
        self.kdtree = KDTree(np.column_stack((track['x'], track['y'])))
        self.nx, self.nu, self.T = 8, 3, HORIZON
        self._x_warm = None
        self._u_warm = None
        self._build_dynamics()
        self._build()

    def _build_dynamics(self):
        p = self.p
        Caf, Car, m, Lf, Lr, Iz = p['Caf'], p['Car'], p['m'], p['Lf'], p['Lr'], p['Iz']
        Ix, h, g = p['Ix'], p['h'], p['g']
        z = ca.SX.sym('z', self.nx)
        u = ca.SX.sym('u', self.nu)
        psi, beta, dpsi, v, theta, dtheta = z[2], z[3], z[4], z[5], z[6], z[7]
        a_x, Mz, M_roll = u[0], u[1], u[2]
        v_eps = ca.sqrt(v**2 + 0.2)
        dbeta = beta * (-(Caf + Car) / (m * v_eps)) \
            + dpsi * ((-Caf * Lf + Car * Lr) / (m * v_eps**2) - 1)
        ddpsi = beta * ((-Caf * Lf + Car * Lr) / Iz) \
            - dpsi * ((Caf * Lf**2 + Car * Lr**2) / (Iz * v_eps)) + Mz / Iz
        a_y = v_eps * (dbeta + dpsi)
        ddtheta = (-m * g * h * ca.sin(theta)
                   - m * a_y * h * ca.cos(theta) + M_roll) / Ix
        dz = ca.vertcat(v * ca.cos(psi + beta), v * ca.sin(psi + beta),
                        dpsi, dbeta, ddpsi, a_x, dtheta, ddtheta)
        fc = ca.Function('fc', [z, u], [dz])

        dt_sub = DT_MPC / N_SUBSTEP
        zk = z
        for _ in range(N_SUBSTEP):
            k1 = fc(zk, u)
            k2 = fc(zk + 0.5 * dt_sub * k1, u)
            k3 = fc(zk + 0.5 * dt_sub * k2, u)
            k4 = fc(zk + dt_sub * k3, u)
            zk = zk + (dt_sub / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.fd = ca.Function('fd', [z, u], [zk])

    def _build(self):
        opti = ca.Opti()
        opti.solver('ipopt',
            {"expand": True, "print_time": 0, "verbose": False, "error_on_fail": 0},
            {"max_iter": 80, "print_level": 0, "tol": 1e-3,
             "acceptable_tol": 5e-2, "acceptable_iter": 3,
             "warm_start_init_point": "yes"})

        x = opti.variable(self.nx, self.T + 1)
        u = opti.variable(self.nu, self.T)
        x0 = opti.parameter(self.nx)
        xr = opti.parameter(self.T + 1)
        yr = opti.parameter(self.T + 1)
        pr = opti.parameter(self.T + 1)
        vr = opti.parameter(self.T + 1)
        w_lat = opti.parameter(self.T + 1)
        w_head = opti.parameter(self.T + 1)

        opti.subject_to(x[:, 0] == x0)
        for k in range(self.T):
            opti.subject_to(x[:, k+1] == self.fd(x[:, k], u[:, k]))

        cost = 0
        R = ca.DM(R_CTRL)
        Rd = ca.DM(R_RATE)
        for k in range(self.T + 1):
            ex = x[0, k] - xr[k]
            ey = x[1, k] - yr[k]
            cost += w_lat[k] * (ex**2 + ey**2)
            cost += w_head[k] * (1 - ca.cos(x[2, k] - pr[k]))
            cost += W_SPEED * (x[5, k] - vr[k])**2

            cte = -ca.sin(pr[k]) * ex + ca.cos(pr[k]) * ey
            cost += W_BOUNDS * ca.fmax(0, ca.fabs(cte) - CTE_SOFT_M)**2

            beta_k = x[3, k]
            cost += W_BETA * (ca.fmax(0, beta_k - BETA_MAX)**2 +
                              ca.fmax(0, -beta_k - BETA_MAX)**2)

            dpsi_k = x[4, k]
            wz_lim = WZ_MAX * ca.fmin(1.0, x[5, k] / V_TURN)
            cost += W_YAW * (ca.fmax(0, dpsi_k - wz_lim)**2 +
                             ca.fmax(0, -dpsi_k - wz_lim)**2)

            theta_k = x[6, k]
            cost += W_LTR * (x[5, k] * dpsi_k + G * theta_k)**2
            cost += W_ROLL_BOUND * (ca.fmax(0, theta_k - ROLL_MAX)**2 +
                                    ca.fmax(0, -theta_k - ROLL_MAX)**2)

            opti.subject_to(x[5, k] >= self.v_min)
            opti.subject_to(x[5, k] <= self.v_max)

            if k < self.T:
                opti.subject_to(u[0, k] >= -AX_MAX)
                opti.subject_to(u[0, k] <= AX_MAX)
                opti.subject_to(u[2, k] >= -M_ROLL_MAX)
                opti.subject_to(u[2, k] <= M_ROLL_MAX)
                cost += u[:, k].T @ R @ u[:, k]
                if k >= 1:
                    du = u[:, k] - u[:, k-1]
                    cost += du.T @ Rd @ du

        opti.minimize(cost)

        self._opti = opti
        self._x = x
        self._u = u
        self._x0 = x0
        self._xr = xr
        self._yr = yr
        self._pr = pr
        self._vr = vr
        self._w_lat = w_lat
        self._w_head = w_head

    def _arc_length(self, x_pos, y_pos):
        _, idx = self.kdtree.query([float(x_pos), float(y_pos)])
        t = self.track
        dx = float(x_pos) - t['x'][idx]
        dy = float(y_pos) - t['y'][idx]
        ds = dx * np.cos(t['theta'][idx]) + dy * np.sin(t['theta'][idx])
        return (t['s'][idx] + ds) % t['L']

    def solve(self, x0_arr):
        x0_safe = np.array(x0_arr, dtype=float).copy()
        x0_safe[5] = max(x0_safe[5], self.v_min)
        self._opti.set_value(self._x0, x0_safe)

        s = self._arc_length(x0_safe[0], x0_safe[1])
        t = self.track
        L = t['L']
        xrv = np.zeros(self.T + 1)
        yrv = np.zeros(self.T + 1)
        prv = np.zeros(self.T + 1)
        vrv = np.zeros(self.T + 1)
        w_lat = np.full(self.T + 1, W_LATERAL, dtype=float)
        w_head = np.full(self.T + 1, W_HEADING, dtype=float)
        for k in range(self.T + 1):
            sw = s % L
            xrv[k] = np.interp(sw, t['s'], t['x'])
            yrv[k] = np.interp(sw, t['s'], t['y'])
            prv[k] = np.interp(sw, t['s'], t['theta'])
            base_v = float(np.interp(sw, t['s'], t['v']))
            vrv[k] = float(np.clip(base_v, self.v_min, self.v_max))
            s += vrv[k] * DT_MPC

        self._opti.set_value(self._xr, xrv)
        self._opti.set_value(self._yr, yrv)
        self._opti.set_value(self._pr, prv)
        self._opti.set_value(self._vr, vrv)
        self._opti.set_value(self._w_lat, w_lat)
        self._opti.set_value(self._w_head, w_head)

        if self._x_warm is None:
            self._opti.set_initial(self._x, x0_safe.reshape((self.nx, 1)) @ np.ones((1, self.T + 1)))
            self._opti.set_initial(self._u, np.zeros((self.nu, self.T)))
        else:
            self._opti.set_initial(self._x, self._x_warm)
            self._opti.set_initial(self._u, self._u_warm)

        try:
            sol = self._opti.solve()
            x_opt = sol.value(self._x)
            u_opt = sol.value(self._u)
        except Exception:
            try:
                x_opt = self._opti.debug.value(self._x)
                u_opt = self._opti.debug.value(self._u)
                if not (np.all(np.isfinite(x_opt)) and np.all(np.isfinite(u_opt))):
                    self._x_warm = None
                    return None, None, False
            except Exception:
                self._x_warm = None
                return None, None, False

        self._x_warm = np.hstack((x_opt[:, 1:], x_opt[:, -1:]))
        self._u_warm = np.hstack((u_opt[:, 1:], u_opt[:, -1:]))
        return x_opt, u_opt, True


TRAIN_AX = 2.0
TRAIN_MPSI = 8.0
TRAIN_MTHETA = 15.0

class MomentMPC:

    def __init__(self, raceline=DEFAULT_RACELINE, v_max=V_MAX, tilt=True):
        track = _load_track(raceline)
        self.mpc = _BicycleMPC(track, v_max=v_max)
        self.track_length = track['L']
        self.tilt = tilt
        self._last_psi = None

    def plan(self, x, y, psi, v, phi=0.0):
        if self._last_psi is None:
            dpsi = 0.0
        else:
            dpsi = ((psi - self._last_psi + np.pi) % (2 * np.pi) - np.pi) / DT_MPC
        self._last_psi = psi

        x0 = np.array([x, y, psi, 0.0, dpsi, max(v, V_MIN), phi, 0.0])
        x_opt, u_opt, ok = self.mpc.solve(x0)

        t = self.mpc.track
        s = self.mpc._arc_length(x, y)
        v_ref = float(np.clip(np.interp(s % t['L'], t['s'], t['v']), V_MIN, self.mpc.v_max))

        if ok:
            a_x = float(np.clip(u_opt[0, 0], -TRAIN_AX, TRAIN_AX))
            m_psi = float(np.clip(u_opt[1, 0], -TRAIN_MPSI, TRAIN_MPSI))
            m_theta = float(np.clip(u_opt[2, 0], -TRAIN_MTHETA, TRAIN_MTHETA)) if self.tilt else 0.0
            command = np.array([a_x, m_psi, m_theta], dtype=np.float32)
            theta_des = float(x_opt[6, 1])
        else:
            command = np.zeros(3, dtype=np.float32)
            theta_des = 0.0
        return command, {"v_ref": v_ref, "s": s, "theta_des": theta_des}
