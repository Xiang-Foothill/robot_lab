# Go2W MuJoCo track-following sim

Standalone (ROS-free) MuJoCo rig to validate the smooth-steer RL model following the
raceline with the new `[lin_acc_x, roll_rate, ang_vel_z]` command interface, before
deploying to the robot.

## Files
- `gen_go2w_mjcf.py` — generates `assets/go2w.xml` from the robot_lab Go2W URDF (converts
  `.dae`→`.stl`, floating base, ground plane, IsaacLab-matched actuators, IMU/gyro sensor).
  Run once: `python gen_go2w_mjcf.py`.
- `accel_pursuit_planner.py` — `AccelPursuitPlanner`: raceline + pose → `[ax, 0, wz]`.
  Acceleration variant of `pure_pursuit_node`; outer speed loop `ax = Kp·(v_ref − v)`,
  yaw rate from pure-pursuit geometry. Pure NumPy, drops into a ROS node that publishes
  an `action_seq` of shape `(HORIZON, 3)` via `plan_horizon()`.
- `policy_runner.py` — `PolicyRunner`: the exact 54-d obs layout and 16-d action decode of
  the IsaacLab env. This is the piece to port into the robot `rl_node`.
- `run_mujoco_track_sim.py` — the test loop (50 Hz control, 10 Hz planning, 200 Hz physics).
- `isaaclab_command_check.py` — forces a fixed command in native IsaacLab to ground-truth a
  policy's command response.

## Run
```
conda activate env_isaaclab_v2
python gen_go2w_mjcf.py
python run_mujoco_track_sim.py --duration 30            # headless, saves track_result.png
python run_mujoco_track_sim.py --viewer                 # interactive viewer
python isaaclab_command_check.py --ax 1.0 --wz 1.0      # native-sim ground truth
```

## Policy interface (smooth-steer)
- obs (54): `base_ang_vel·0.25 (3)`, `command (3) = [ax, roll_rate, wz]`,
  `joint_pos_rel (16, wheels zeroed)`, `joint_vel·0.05 (16)`, `last_action (16)`.
- action (16): 12 leg position deltas (`q = default + scale·a`, scale hip 0.125 / others 0.25)
  + 4 wheel velocity targets (`dq = 5.0·a`).
- joint order `FR,FL,RR,RL` legs then `FR,FL,RR,RL` wheels; command index 2 = yaw rate.
- actuators match IsaacLab: legs position kp=70 / damping 10, wheels velocity kv=0.5,
  torque limited to ±23.5 N·m (the effort limit used in training).

## Deployment mapping (robot)
The planner publishes `action_seq (HORIZON, 3)` of `[ax, 0, wz]`; the robot `rl_node`
selects the horizon index by elapsed time, fills `velocity_commands`, and runs `PolicyRunner`
(obs from IMU gyro + joint states, action → leg `q` / wheel `dq`). Same topology as the
current velocity-tracking deployment, with the command contract changed to acceleration.

## Finding
The trained model `model_13998` does **not** steer: for a commanded yaw rate of 1.0 rad/s it
achieves ~0.01 rad/s — confirmed in both this MuJoCo rig and the native IsaacLab env (verified
with `isaaclab_command_check.py`), so it is a property of the model, not a sim discrepancy.
Forward acceleration tracking works (the robot drives but cannot turn onto the track). The
likely cause is reward balance: yaw-tracking weight (1.5) is dominated by the pose/orientation
penalties (`joint_pos_penalty` −8, `flat_orientation_l2` −5, `base_height_l2` −3, `action_sync`
−2). The model needs retraining with a stronger yaw incentive before trajectory following is
viable; this rig and `isaaclab_command_check.py` can verify the next model before robot tests.
