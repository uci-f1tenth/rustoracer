# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "rustoracerpy",
#     "rerun-sdk>=0.20.0",
#     "tensordict>=0.11.0",
#     "torch>=2.10.0",
#     "tqdm>=4.67.3",
#     "tyro>=1.0.8",
#     "wandb[media]>=0.25.0",
#     "eventlet==0.33.3",
#     "flask==1.1.1",
#     "flask-socketio==4.1.0",
#     "gevent-websocket==0.10.1",
#     "itsdangerous==2.0.1",
#     "jinja2==3.0.3",
#     "numpy>=2.2.6",
#     "pillow>=11.3.0",
#     "python-engineio==3.13.0",
#     "python-socketio==4.2.0",
#     "werkzeug==2.0.3",
# ]
#
# [tool.uv.sources]
# rustoracerpy = { path = ".." }
# ///
"""Rustoracer PPO agent → AutoDrive F1TENTH bridge.

The trained model outputs [d_steer, torque] both in [-1, 1]:
  - d_steer: normalized steering velocity (±1 → ±3.2 rad/s)
  - torque:  normalized drive torque (±1 → full brake/throttle)

AutoDRIVE accepts:
  - steering_command in [-1, 1] → maps to [-0.5236, 0.5236] rad (±30°)
  - throttle_command in [-1, 1] → maps to full reverse/forward torque

We integrate d_steer into a tracked steering angle (matching the sim)
and send it as a normalized steering command.

Observation vector (obs_dim = N_LIDAR_OBS + 3 + 2 + N_LOOK*2):
  [0:N_LIDAR_OBS]               downsampled lidar ranges
  [N_LIDAR_OBS]                 speed (m/s)
  [N_LIDAR_OBS+1]               steering angle (rad)
  [N_LIDAR_OBS+2]               yaw rate (rad/s)
  [N_LIDAR_OBS+3]               heading difference to centerline (rad)
  [N_LIDAR_OBS+4]               lateral offset from centerline (m)
  [N_LIDAR_OBS+5 : obs_dim]     N_LOOK lookahead waypoints in car frame (dx, dy each)
"""

import argparse

import numpy as np
import rerun as rr
import torch
import torch.nn as nn
import socketio
import eventlet
from flask import Flask
import autodrive
from rustoracerpy.rustoracer import PySim

# ── CLI ──
_parser = argparse.ArgumentParser()
_parser.add_argument("--map", default="maps/my_map.yaml", help="path to map YAML")
_parser.add_argument(
    "--checkpoint", default="checkpoints/agent_final.pt", help="path to checkpoint"
)
_parser.add_argument(
    "--rerun", action="store_true", help="stream debug data to a Rerun viewer"
)
_cli = _parser.parse_args()

if _cli.rerun:
    rr.init("rustoracer/autodrive_bridge", spawn=True)

# ── Vehicle constants (from AutoDRIVE docs / sim) ──
STEER_MAX = 0.5236  # rad (±30°)
STEER_VEL_MAX = 3.2  # rad/s  (steering rate limit)
N_LIDAR_RAW = 1080  # AutoDRIVE LIDAR measurements per scan
N_LIDAR_OBS = 108  # downsampled beams to match sim training
N_LOOK = 10  # lookahead waypoints (must match training; see sim.rs N_LOOK)

# ── Load checkpoint ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(_cli.checkpoint, map_location=DEVICE, weights_only=False)
obs_mean, obs_var = ckpt["obs_rms_mean"].numpy(), ckpt["obs_rms_var"].numpy()
OBS_DIM = obs_mean.shape[0]  # inferred from checkpoint (133 for centerline model)


# ── Agent (must match training architecture) ──
class Agent(nn.Module):
    def __init__(self, obs=OBS_DIM, act=2, h=256):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, act)
        )

    def forward(self, x):
        return self.actor_mean(x)


agent = Agent()
agent.load_state_dict(ckpt["model"], strict=False)
agent.to(DEVICE).eval()

# ── Centerline setup (mirrors sim.rs skeleton / look_step logic) ──
_sim = PySim(_cli.map, 1, 1)  # num_envs=1, max_steps=1 — only used to extract the skeleton
_skeleton_pts: np.ndarray = _sim.skeleton.reshape(-1, 2)  # (N, 2) world coords
del _sim

N_WPS = len(_skeleton_pts)
_diffs = np.hypot(
    np.roll(_skeleton_pts[:, 0], -1) - _skeleton_pts[:, 0],
    np.roll(_skeleton_pts[:, 1], -1) - _skeleton_pts[:, 1],
)
LOOK_STEP = max(1, round(1.0 / float(_diffs.mean())))

# LiDAR geometry constants (Hokuyo UST-10LX as used by AutoDRIVE: 270° FOV)
LIDAR_IDX = np.round(np.linspace(0, N_LIDAR_RAW - 1, N_LIDAR_OBS)).astype(int)
_LIDAR_FOV = np.deg2rad(270.0)
_LIDAR_ANGLES_RAW = np.linspace(-_LIDAR_FOV / 2, _LIDAR_FOV / 2, N_LIDAR_RAW)
_LIDAR_ANGLES_OBS = _LIDAR_ANGLES_RAW[LIDAR_IDX]  # angles for the 108 downsampled beams

if _cli.rerun:
    # Log the full centerline once as a static entity
    rr.log("world/centerline", rr.LineStrips2D([_skeleton_pts]), static=True)


def _yaw_from_quat(q: np.ndarray) -> float:
    """Extract yaw angle from a [qx, qy, qz, qw] quaternion."""
    qx, qy, qz, qw = q
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


# ── Helpers ──
current_steer = 0.0  # tracked steering angle [rad] (for command integration)
NOMINAL_DT = 1.0 / 60.0  # fixed dt for steering integration [s]
_rr_step = 0  # Rerun frame counter


def get_obs(f):
    """Build the OBS_DIM-d observation vector.

    Includes downsampled lidar, vehicle states (speed, steering, yaw rate),
    and centerline features (heading diff, lateral offset, N_LOOK lookahead
    waypoints in the car's local frame) to match the sim training environment.

    Returns:
        obs_norm: normalized observation array fed to the model
        debug:    dict of raw intermediate values for visualization
    """
    # ── LiDAR ──
    lidar = f.lidar_range_array[LIDAR_IDX]

    # ── Vehicle states ──
    vel = f.speed
    steer_rad = f.steering * STEER_MAX
    yaw_rate = f.angular_velocity[2]

    # ── Car pose ──
    x, y = f.position[0], f.position[1]
    theta = _yaw_from_quat(f.orientation_quaternion)
    ch, sh = np.cos(theta), np.sin(theta)

    # ── Nearest centerline waypoint ──
    wi = int(np.argmin(np.hypot(_skeleton_pts[:, 0] - x, _skeleton_pts[:, 1] - y)))

    # Centerline heading at wi
    cx, cy = _skeleton_pts[wi]
    nx, ny = _skeleton_pts[(wi + 1) % N_WPS]
    cth = np.arctan2(ny - cy, nx - cx)

    # Heading difference (wrapped to [-π, π])
    heading_diff = ((cth - theta) + np.pi) % (2.0 * np.pi) - np.pi

    # Signed lateral offset: project the vector from the centerline point to the car
    # onto the centerline's left-normal direction (perpendicular, rotated 90° CCW).
    lat_off = -(x - cx) * np.sin(cth) + (y - cy) * np.cos(cth)

    # ── Lookahead waypoints in car frame ──
    lookahead = np.empty(N_LOOK * 2)
    for k in range(N_LOOK):
        wx, wy = _skeleton_pts[(wi + (k + 1) * LOOK_STEP) % N_WPS]
        dx, dy = wx - x, wy - y
        lookahead[k * 2] = dx * ch + dy * sh
        lookahead[k * 2 + 1] = -dx * sh + dy * ch

    obs = np.concatenate(
        [lidar, [vel, steer_rad, yaw_rate, heading_diff, lat_off], lookahead]
    )
    debug = dict(
        lidar_raw=lidar.copy(),
        x=float(x), y=float(y), theta=float(theta),
        vel=float(vel), steer_rad=float(steer_rad), yaw_rate=float(yaw_rate),
        heading_diff=float(heading_diff), lat_off=float(lat_off),
        wi=wi, cx=float(cx), cy=float(cy),
    )
    return np.clip((obs - obs_mean) / np.sqrt(obs_var + 1e-8), -10, 10), debug


def _log_rerun(dbg, d_steer_raw, torque_raw, torque_cmd, tracked_steer, steer_cmd):
    """Stream one frame of debug data to the Rerun viewer."""
    global _rr_step
    rr.set_time_sequence("frame", _rr_step)
    _rr_step += 1

    x, y, theta = dbg["x"], dbg["y"], dbg["theta"]
    lidar_raw = dbg["lidar_raw"]
    wi, cx, cy = dbg["wi"], dbg["cx"], dbg["cy"]

    # ── World-space geometry ──
    rr.log("world/car", rr.Points2D([[x, y]], radii=[0.15], colors=[[43, 127, 255]]))
    rr.log(
        "world/heading",
        rr.Arrows2D(
            origins=[[x, y]],
            vectors=[[0.5 * np.cos(theta), 0.5 * np.sin(theta)]],
            colors=[[255, 255, 0]],
        ),
    )
    rr.log("world/nearest_wp", rr.Points2D([[cx, cy]], radii=[0.1], colors=[[255, 165, 0]]))

    lk_world = np.array(
        [_skeleton_pts[(wi + (k + 1) * LOOK_STEP) % N_WPS] for k in range(N_LOOK)]
    )
    rr.log("world/lookahead", rr.Points2D(lk_world, radii=[0.08], colors=[[0, 255, 0]]))

    # LiDAR rays in world frame (Hokuyo UST-10LX: 270° FOV centred on forward axis)
    ray_ends = np.c_[
        x + lidar_raw * np.cos(theta + _LIDAR_ANGLES_OBS),
        y + lidar_raw * np.sin(theta + _LIDAR_ANGLES_OBS),
    ]
    origins = np.broadcast_to([[x, y]], ray_ends.shape).copy()
    rr.log(
        "world/lidar",
        rr.LineStrips2D(np.stack([origins, ray_ends], axis=1), colors=[[255, 80, 80]]),
    )

    # ── Scalar observations ──
    rr.log("obs/speed_mps", rr.Scalar(dbg["vel"]))
    rr.log("obs/steer_rad", rr.Scalar(dbg["steer_rad"]))
    rr.log("obs/yaw_rate_rads", rr.Scalar(dbg["yaw_rate"]))
    rr.log("obs/heading_diff_rad", rr.Scalar(dbg["heading_diff"]))
    rr.log("obs/lat_offset_m", rr.Scalar(dbg["lat_off"]))

    # Raw LiDAR ranges as a bar chart (easy to spot clipping / short returns)
    rr.log("obs/lidar_ranges", rr.BarChart(lidar_raw.astype(np.float32)))

    # ── Model outputs and sent commands ──
    rr.log("actions/d_steer_raw", rr.Scalar(d_steer_raw))
    rr.log("actions/torque_raw", rr.Scalar(torque_raw))
    rr.log("actions/tracked_steer_rad", rr.Scalar(tracked_steer))
    rr.log("actions/steering_command", rr.Scalar(steer_cmd))
    rr.log("actions/throttle_command", rr.Scalar(torque_cmd))


# ── AutoDrive bridge ──
f1 = autodrive.RoboRacer()
f1.id = "V1"
sio = socketio.Server()
app = Flask(__name__)


@sio.on("connect")
def connect(sid, environ):
    print("Connected!")


@sio.on("Bridge")
def bridge(sid, data):
    global current_steer
    if not data:
        return
    f1.parse_data(data)

    obs_norm, dbg = get_obs(f1)
    with torch.no_grad():
        act = agent(
            torch.tensor(obs_norm, device=DEVICE, dtype=torch.float32).unsqueeze(0)
        )
    a = act.cpu().numpy().flatten().clip(-1, 1)

    # a[0] = d_steer ∈ [-1, 1] → steering velocity
    # a[1] = torque   ∈ [-1, 1] → throttle command
    d_steer = float(a[0])
    torque = float(a[1]) * 0.03

    # Integrate steering velocity with a fixed dt (matches sim's fixed-rate control)
    sv = d_steer * STEER_VEL_MAX  # rad/s
    current_steer = np.clip(current_steer + sv * NOMINAL_DT, -STEER_MAX, STEER_MAX)

    # Send to AutoDRIVE: steering_command in [-1, 1], throttle_command in [-1, 1]
    f1.steering_command = current_steer / STEER_MAX  # normalize to [-1, 1]
    f1.throttle_command = torque

    if _cli.rerun:
        _log_rerun(dbg, d_steer, float(a[1]), torque, current_steer, f1.steering_command)

    try:
        sio.emit("Bridge", data=f1.generate_commands())
    except Exception as e:
        print(e)


if __name__ == "__main__":
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
