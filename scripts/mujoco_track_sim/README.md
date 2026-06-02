# Go2W MuJoCo track-following sim

Standalone (ROS-free) MuJoCo rig to validate the Go2W smooth-steer RL policy following
the raceline before deploying to the robot. The policy is trained in IsaacLab with the
`[lin_vel_x, roll_rate, ang_vel_z]` differential command; this rig reproduces the exact
observation/action interface so a policy that works here can be ported to the robot
`rl_node` unchanged.

## Files
- `gen_go2w_mjcf.py` — generates `assets/go2w.xml` from the robot_lab Go2W URDF (converts
  `.dae`→`.stl`, floating base, ground plane, IsaacLab-matched actuators, IMU/gyro sensor).
  Run once after a URDF change. Key constants: `LEG_KP=70`, `LEG_KD=10`, `WHEEL_KV=0.5`,
  `EFFORT_LIMIT=23.5`, `BASE_INIT_HEIGHT=0.45`, `GROUND_FRICTION=1.1` (matches the lab floor).
- `policy_runner.py` — `PolicyRunner`: the exact 54-d observation layout and 16-d action
  decode of the IsaacLab env. This is the piece to port into the robot `rl_node`.
- `accel_pursuit_planner.py` — `AccelPursuitPlanner`: raceline + pose → `[vx, 0, wz]`.
  Passes the raceline reference speed directly as a velocity target; yaw from pure-pursuit
  geometry; roll fixed at 0. Drops into a ROS node that publishes an `action_seq` of shape
  `(HORIZON, 3)` via `plan_horizon()`.
- `run_mujoco_track_sim.py` — the test loop (50 Hz control, 10 Hz planning, 200 Hz physics).
  Defaults to the current best policy; override with `--policy`.
- `assets/go2w.xml`, `assets/meshes/` — generated MuJoCo model (do not hand-edit; regenerate).

## Run
```bash
conda activate isaac
cd ~/robot_lab/scripts/mujoco_track_sim

python gen_go2w_mjcf.py                         # regenerate the MJCF (only after URDF changes)
python run_mujoco_track_sim.py --viewer         # interactive viewer (real-time, camera follows robot)
python run_mujoco_track_sim.py --duration 60    # headless, saves track_result.png
python run_mujoco_track_sim.py --policy <path/to/exported/policy.pt> --viewer
```
Useful flags: `--policy <path>`, `--duration <sec>`, `--raceline <npz>`, `--no_real_time`
(unthrottle the viewer), `--plot <png>`.

## Policy interface (smooth-steer)
- obs (54): `base_ang_vel·0.25 (3)`, `command (3) = [lin_vel_x, roll_rate, ang_vel_z]`,
  `joint_pos_rel (16, wheels zeroed)`, `joint_vel·0.05 (16)`, `last_action (16)`.
- action (16): 12 leg position deltas (`q = default + scale·a`, scale hip 0.125 / others 0.25)
  + 4 wheel velocity targets (`dq = 5.0·a`).
- joint order `FR,FL,RR,RL` legs then `FR,FL,RR,RL` wheels; command index 2 = yaw rate.
- actuators match IsaacLab: legs position kp=70 / damping 10, wheels velocity kv=0.5,
  torque limited to ±23.5 N·m (the effort limit used in training — do not inflate the wheel
  gain to force tracking, it overstates torque/overload).

## Command space change: acceleration → velocity
The policy was retrained (2026-06) to accept `lin_vel_x` (m/s) instead of `lin_acc_x` (m/s²)
as command index 0. The training range is still `(-2.0, 2.0)`.

**Why:** acceleration commands do not pin steady-state speed — at constant cruise the
commanded value earns little reward and travel direction is left to the optimizer, which can
settle in a backward-cruising basin. Velocity commands make the policy act as a closed-loop
speed tracker: `vx=0` is an explicit hold-position instruction, and `rel_standing_envs=0.15`
trains this explicitly. The reward kernel std was tightened from 1.0 → 0.25 for more precise
tracking.

**Planner side:** planners now pass reference velocity directly (`vx = v_ref`); the MPC
publishes `x_opt[5, k+1]` (planned speed from the bicycle model) instead of `u_opt[0, k]`
(ax). Internally the MPC still optimizes ax — only the extracted output changed.

## Exporting a policy to test
`play.py` writes `<run>/exported/policy.pt` (TorchScript) on load, which this rig consumes.
Always pin the run with `--load_run` — a stray `unitree_go2w_smooth_steer/unitree_go2w_flat/`
folder sorts after the date dirs and gets auto-selected otherwise:
```bash
python ~/robot_lab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-SmoothSteer-Unitree-Go2W-v0 \
  --headless --num_envs 1 --load_run <YYYY-MM-DD_HH-MM-SS>
```

## Deployment mapping (robot)
The planner publishes `action_seq (HORIZON, 3)` of `[vx, 0, wz]`; the robot `rl_node` selects
the horizon index by elapsed time, fills `velocity_commands`, and runs `PolicyRunner` (obs
from IMU gyro + joint states, action → leg `q` / wheel `dq`). Same topology as the previous
deployment, with the command contract changed from acceleration to velocity.

## Context / validated findings
- The rig is **validated against native IsaacLab**: forced-command tests match, so MuJoCo
  results here are trustworthy.
- **Frame convention:** body +x is visual-forward (front hips at +0.1934 m). Positive
  `lin_vel_x` drives forward.
- **Tracking tightness is speed-limited, not model-limited**: with yaw capped at π/3 rad/s,
  turn radius ≈ v / yaw, so lowering the planner target speed tightens the line on tight tracks.
  The planner caps target speed at `V_MAX = 1.5` m/s; on this raceline that keeps mean
  cross-track error ~0.05 m. Raise it for faster, wider laps.
- **After any retrain, validate the `vx`→direction mapping in MuJoCo before track tests.**
  Check that `vx > 0` drives forward and `vx = 0` holds position with no backward drift.
