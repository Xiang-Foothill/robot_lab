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
- `policy_runner.py` — `PolicyRunner`: 57-d observation layout, 16-d action decode, and
  calf lean overlay for active tilt. Ported verbatim into the robot `accel_rl_node`.
- `accel_pursuit_planner.py` — `AccelPursuitPlanner`: raceline + pose → `[vx, roll, wz]`.
  Pure-pursuit geometry; supports both `.npz` and `.h5` raceline files.
- `run_mujoco_track_sim.py` — the test loop (50 Hz control, 10 Hz planning, 200 Hz physics).
- `assets/go2w.xml`, `assets/meshes/` — generated MuJoCo model (do not hand-edit; regenerate).

## Run
```bash
conda activate go2w_race_py38
cd ~/robot_lab/scripts/mujoco_track_sim

python gen_go2w_mjcf.py                   # regenerate MJCF (only after URDF changes)
python run_mujoco_track_sim.py --viewer   # interactive viewer, real-time
python run_mujoco_track_sim.py --duration 60 --no_real_time  # headless, saves track_result.png

# mpc planner (recommended)
python run_mujoco_track_sim.py --planner mpc --viewer

# active tilt comparison
python run_mujoco_track_sim.py --planner mpc --viewer               # no tilt (control group)
python run_mujoco_track_sim.py --planner mpc --tilt --viewer        # active tilt enabled

# specify policy explicitly (required when default policy path changes)
python run_mujoco_track_sim.py --planner mpc --tilt \
  --policy ~/go2w_planner_ws/data/<run>/exported/policy.pt --viewer
```
Useful flags: `--planner {pursuit,mpc}`, `--tilt`, `--policy <path>`, `--raceline <path>`,
`--duration <sec>`, `--no_real_time`, `--v_max <float>`, `--plot <png>`.

## Raceline
The default raceline is:
```
~/Go2w_race/Go2WRace/go2w_controllers/planners/trajectory/casadi_dynamics_bicycle_trajectory_vx_3.2.h5
```
This is the correct lab-scale track (L ≈ 14.3 m). Both `.h5` and `.npz` files are supported.
The old `.npz` files in `go2w_planner_ws/data/` were generated for a larger track and should
not be used.

## Policy interface (smooth-steer)
- obs (57): `base_lin_vel·1.0 (3)`, `base_ang_vel·0.25 (3)`, `command (3) = [lin_vel_x, roll_rate, ang_vel_z]`,
  `joint_pos_rel (16, wheels zeroed)`, `joint_vel·0.05 (16)`, `last_action (16)`.
- action (16): 12 leg position deltas (`q = default + scale·a`, scale hip 0.125 / others 0.25)
  + 4 wheel velocity targets (`dq = 5.0·a`).
- joint order `FR,FL,RR,RL` legs then `FR,FL,RR,RL` wheels.
- actuators match IsaacLab: legs position kp=70 / damping 10, wheels velocity kv=0.5,
  torque limited to ±23.5 N·m (do not inflate — overstates torque/overcurrent).

## Active tilt
The `--tilt` flag enables a model-based calf lean overlay that physically banks the body
into corners without requiring a retrained policy.

**Mechanism:** hip abduction (the "shoulder" joint) only shifts wheel stance width — it
produces zero body roll against 4 grounded contact points. The only joint that changes leg
height is the calf (knee). Asymmetric calf flexion (right calves flex more → right side
shortens → body leans left) produces ~1:1 rad body roll per rad calf offset, calibrated
in simulation.

**Implementation:** `PolicyRunner._lean_default(lean_angle)` shifts the calf joint default
that `joint_pos_rel` is computed against, so the policy sees the lean pose as neutral and
does not fight the overlay. The bank angle target is the LTR=0 condition from
`unitree_racing`: `θ_des = -atan(v·ψ̇/g)`. The MPC planner passes this through `theta_des`
in the plan info dict; the sim loop applies it to both obs and action decode.

**Results (flat policy, 40 s, 3 laps):**

| | no tilt | tilt |
|---|---|---|
| cross-track error | 0.062 m mean, 0.135 m max | 0.065 m mean, 0.166 m max |
| speed | 1.08 m/s mean | 1.06 m/s mean |
| body roll delivered | 1° | 9° (vs 14° target, 64% delivery) |

Partial delivery (64%) is due to the flat policy's symmetric PD control partially resisting
the calf offset. Tracking is preserved. The gain `K_CALF_LEAN` in `policy_runner.py` can
be increased (try 1.3–1.5) to push delivered roll closer to target.

**Robot deployment:** set `-p tilt_enable:=true` on both `fw_mpc_node` (workstation) and
`accel_rl_node` (robot). Both default to off for clean tilt-vs-no-tilt comparisons.

## Exporting a policy to test
`play.py` writes `<run>/exported/policy.pt` (TorchScript) on load, which this rig consumes.
Pin the run with `--load_run` — a stray `unitree_go2w_flat/` folder sorts after the date
dirs and gets auto-selected otherwise:
```bash
python ~/robot_lab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-SmoothSteer-Unitree-Go2W-v0 \
  --headless --num_envs 1 --load_run <YYYY-MM-DD_HH-MM-SS>
```

## Tilt policy training
A separate task `RobotLab-Isaac-Velocity-Tilt-Unitree-Go2W-v0` is registered for training
a policy that actively responds to roll_rate commands. The no-tilt smooth-steer task is
kept frozen as the control group. Train with:
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Tilt-Unitree-Go2W-v0 \
  --resume --load_run <smooth-steer-run-date> \
  --max_iterations 2000 --headless
```
Key cfg changes in `tilt_env_cfg.py` vs smooth-steer: `roll_rate` range opened to ±1.5 rad/s;
`flat_orientation_l2` replaced by pitch-only penalty (roll free); `action_sync` hip groups
split into left/right pairs so the calf mechanism can deliver lean while thighs/wheels stay
symmetric (preserving rigid chassis style).

## Deployment mapping (robot)
The planner publishes `action_seq (HORIZON, 3)` of `[vx, roll_rate, wz]` at 10 Hz. The robot
`accel_rl_node` selects the horizon step by elapsed time and runs `PolicyRunner` at 50 Hz
(obs from IMU gyro + OptiTrack est_state + joint states, action → leg `q` / wheel `dq` at
500 Hz). When `tilt_enable=true`, both nodes independently apply the LTR=0 bank angle:
the workstation fills `roll_rate` in the action sequence; the robot computes `lean_angle`
from measured v and ψ̇ and applies the calf overlay in `accel_rl_node`.

## Validated findings
- **Raceline size matters:** the h5 default is the correct lab-scale track. The old npz
  files were built for a larger track — the robot will go off course if those are used.
- **Hip abduction cannot produce body roll** on a 4-wheeled grounded robot. Calf asymmetry
  is the correct lean mechanism (verified by forced-joint tests in MuJoCo).
- **RL tilt training is harder than expected:** the flat policy's strong joint-pos-penalty
  and 4-way action_sync prevent calf asymmetry even after 4000 fine-tune iterations. The
  direct calf overlay achieves comparable lean without retraining.
- **Frame convention:** body +x is visual-forward. Positive `lin_vel_x` drives forward.
- **After any retrain**, validate `vx > 0` drives forward and `vx = 0` holds position
  with no backward drift before track tests.
