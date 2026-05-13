# RL Training Environment

---

## Setup

| | |
|---|---|
| **Simulator** | Isaac Sim 5.1.0 (NVIDIA PhysX, GPU) |
| **Framework** | Isaac Lab 2.3.2 |
| **Task** | Velocity-commanded locomotion on rough terrain |
| **Robot** | Quadruped (e.g. Unitree A1/Go2) — 12 DOF |
| **Parallel Envs** | 4,096 |
| **Physics Rate** | 200 Hz · Policy Rate: 50 Hz (decimation = 4) |
| **Episode Length** | 20 s |

---

## Observation Space (47-dim)

| Signal | Dim | Noise |
|---|---|---|
| Base lin/ang velocity | 6 | ±0.1 / ±0.2 |
| Projected gravity | 3 | ±0.05 |
| Velocity command (vx, vy, ωz) | 3 | — |
| Joint positions (rel. default) | 12 | ±0.01 rad |
| Joint velocities | 12 | ±1.5 rad/s |
| Last action | 12 | — |
| Height scan (terrain) | optional | ±0.1 |

Critic receives **clean** observations; actor receives **noisy** observations.

---

## Action Space

- **12 joint position targets** via PD control
- Action scaled per joint group (hip: ×0.125, knee/ankle: ×0.25)
- Clipped to [−100, 100] before execution

---

## Reward Function

| Term | Weight | Purpose |
|---|---|---|
| `track_lin_vel_xy` | **+3.0** | XY velocity tracking |
| `track_ang_vel_z` | **+1.5** | Yaw tracking |
| `upright` | +1.0 | Stay upright |
| `joint_torques_l2` | −2.5×10⁻⁵ | Energy efficiency |
| `joint_acc_l2` | −2.5×10⁻⁷ | Smooth motion |
| `lin_vel_z_l2` | −2.0 | No bouncing |
| `undesired_contacts` | −1.0 | Feet only touch ground |
| `action_rate_l2` | −0.01 | Action smoothness |
| `joint_mirror` | −0.05 | Symmetric gait |

---

## Algorithm: PPO (RSL-RL)

| Hyperparameter | Value |
|---|---|
| Rollout steps per env | 24 |
| Total transitions / update | ~98K |
| Mini-batches | 4 |
| Learning rate | 1×10⁻³ (KL-adaptive) |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Clip ε | 0.2 |
| Epochs per update | 5 |
| Network | MLP [512, 256, 128], ELU |
| Training duration | ~20K iterations (480M steps) |

---

## Sim2Real: Domain Randomization

| Parameter | Range |
|---|---|
| Ground friction | 0.3 – 1.0 |
| Base mass perturbation | ±1.0 kg |
| Body mass scale | 0.7× – 1.3× |
| CoM offset | ±0.05 m |
| PD gains (K_p, K_d) | 0.5× – 2.0× |
| External push force | ±10 N (every 10–15 s) |
| Terrain curriculum | levels 0–5, difficulty adapts |
