# /// script
# requires-python = ">=3.10"
# dependencies = [
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
#     "opencv-contrib-python>=4.11.0.86",
#     "pillow>=11.3.0",
#     "python-engineio==3.13.0",
#     "python-socketio==4.2.0",
#     "werkzeug==2.0.3",
# ]
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
_cli = _parser.parse_args()

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


def _yaw_from_quat(q: np.ndarray) -> float:
    """Extract yaw angle from a [qx, qy, qz, qw] quaternion."""
    qx, qy, qz, qw = q
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


# ── Helpers ──
LIDAR_IDX = np.round(np.linspace(0, N_LIDAR_RAW - 1, N_LIDAR_OBS)).astype(int)
current_steer = 0.0  # tracked steering angle [rad] (for command integration)
NOMINAL_DT = 1.0 / 60.0  # fixed dt for steering integration [s]


def get_obs(f):
    """Build the OBS_DIM-d observation vector.

    Includes downsampled lidar, vehicle states (speed, steering, yaw rate),
    and centerline features (heading diff, lateral offset, N_LOOK lookahead
    waypoints in the car's local frame) to match the sim training environment.
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
    return np.clip((obs - obs_mean) / np.sqrt(obs_var + 1e-8), -10, 10)


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

    with torch.no_grad():
        act = agent(
            torch.tensor(get_obs(f1), device=DEVICE, dtype=torch.float32).unsqueeze(0)
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
    try:
        sio.emit("Bridge", data=f1.generate_commands())
    except Exception as e:
        print(e)


if __name__ == "__main__":
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
