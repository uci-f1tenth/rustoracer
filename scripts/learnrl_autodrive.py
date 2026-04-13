# /// script
# requires-python = ">=3.10, <3.12"
# dependencies = [
#     "rustoracerpy",
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
import time

import autodrive
import eventlet
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import socketio
import torch
import torch.nn as nn
from flask import Flask
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
            nn.Linear(obs, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, act),
        )

    def forward(self, x):
        return self.actor_mean(x)


agent = Agent()
agent.load_state_dict(ckpt["model"], strict=False)
agent.to(DEVICE).eval()

# ── Centerline setup (mirrors sim.rs skeleton / look_step logic) ──
_sim = PySim(
    _cli.map, 1, 1
)  # num_envs=1, max_steps=1 — only used to extract the skeleton
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

    Returns (obs_normalized, debug_dict) where debug_dict contains all raw
    intermediate values for Rerun logging.
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

    # Signed lateral offset
    lat_off = -(x - cx) * np.sin(cth) + (y - cy) * np.cos(cth)

    # ── Lookahead waypoints in car frame ──
    lookahead = np.empty(N_LOOK * 2)
    lookahead_world = np.empty((N_LOOK, 2))
    for k in range(N_LOOK):
        wp_idx = (wi + (k + 1) * LOOK_STEP) % N_WPS
        wx, wy = _skeleton_pts[wp_idx]
        lookahead_world[k] = [wx, wy]
        dx, dy = wx - x, wy - y
        lookahead[k * 2] = dx * ch + dy * sh
        lookahead[k * 2 + 1] = -dx * sh + dy * ch

    obs = np.concatenate(
        [lidar, [vel, steer_rad, yaw_rate, heading_diff, lat_off], lookahead]
    )
    obs_norm = np.clip((obs - obs_mean) / np.sqrt(obs_var + 1e-8), -10, 10)

    debug = dict(
        lidar=lidar,
        vel=vel,
        steer_rad=steer_rad,
        autodrive_steering_raw=f.steering,
        yaw_rate=yaw_rate,
        x=x,
        y=y,
        theta=theta,
        nearest_wp_idx=wi,
        centerline_heading=cth,
        heading_diff=heading_diff,
        lat_off=lat_off,
        lookahead_world=lookahead_world,
        lookahead_local=lookahead.reshape(-1, 2),
        obs_raw=obs,
        obs_norm=obs_norm,
    )
    return obs_norm, debug


# ── Rerun init ──
rr.init("autodrive_debug", spawn=True)

# ── Rerun blueprint ──
rr.send_blueprint(
    rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(
                    name="Map / Trajectory",
                    origin="/world",
                    visual_bounds=rrb.VisualBounds2D(
                        x_range=[-5, 25],
                        y_range=[-5, 25],
                    ),
                ),
                rrb.Spatial2DView(
                    name="Car Frame (LiDAR + Lookahead)",
                    origin="/car_frame",
                ),
                row_shares=[2, 1],
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    name="Agent Raw Outputs",
                    origin="/agent",
                ),
                rrb.TimeSeriesView(
                    name="Steering Pipeline",
                    origin="/steering",
                ),
                rrb.TimeSeriesView(
                    name="Commands Sent",
                    origin="/commands",
                ),
                rrb.TimeSeriesView(
                    name="Centerline Features",
                    origin="/centerline",
                ),
                rrb.TimeSeriesView(
                    name="Vehicle State",
                    origin="/vehicle",
                ),
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    name="Obs: LiDAR (normalized)",
                    origin="/obs/lidar_norm",
                ),
                rrb.BarChartView(
                    name="Obs Vector Snapshot",
                    origin="/obs/full_vector",
                ),
            ),
            column_shares=[2, 2, 1],
        )
    )
)

# ── Log static skeleton / centerline (once) ──
skeleton_3d = np.hstack([_skeleton_pts, np.zeros((N_WPS, 1))])  # add z=0
# Close the loop
skeleton_closed = np.vstack([skeleton_3d, skeleton_3d[:1]])
rr.log(
    "world/centerline",
    rr.LineStrips2D(
        [_skeleton_pts.tolist() + [_skeleton_pts[0].tolist()]],
        colors=[[80, 80, 80]],
    ),
    static=True,
)
rr.log(
    "world/centerline_pts",
    rr.Points2D(_skeleton_pts[::LOOK_STEP], colors=[[50, 50, 50]], radii=0.03),
    static=True,
)

print(f"[rerun] OBS_DIM={OBS_DIM}, N_WPS={N_WPS}, LOOK_STEP={LOOK_STEP}")
print(f"[rerun] obs_mean range: [{obs_mean.min():.3f}, {obs_mean.max():.3f}]")
print(f"[rerun] obs_var  range: [{obs_var.min():.6f}, {obs_var.max():.6f}]")

# ── Track timing ──
_frame_count = 0
_prev_time = None

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
    global current_steer, _frame_count, _prev_time
    if not data:
        return
    f1.parse_data(data)

    # ── Measure real dt ──
    now = time.monotonic()
    real_dt = (now - _prev_time) if _prev_time is not None else NOMINAL_DT
    _prev_time = now

    # ── Set Rerun time ──
    rr.set_time("frame", sequence=_frame_count)
    rr.set_time("wall_clock", timestamp=time.time())
    _frame_count += 1

    # ── Build observation ──
    obs_norm, dbg = get_obs(f1)

    with torch.no_grad():
        act = agent(
            torch.tensor(obs_norm, device=DEVICE, dtype=torch.float32).unsqueeze(0)
        )
    a_raw = act.cpu().numpy().flatten()
    a = a_raw.clip(-1, 1)

    # a[0] = d_steer ∈ [-1, 1] → steering velocity
    # a[1] = torque   ∈ [-1, 1] → throttle command
    d_steer = float(a[0])
    torque = float(a[1]) * 0.03

    # Integrate steering velocity with a fixed dt (matches sim's fixed-rate control)
    sv = d_steer * STEER_VEL_MAX  # rad/s
    steer_before = current_steer
    current_steer = np.clip(current_steer + sv * NOMINAL_DT, -STEER_MAX, STEER_MAX)

    # Send to AutoDRIVE: steering_command in [-1, 1], throttle_command in [-1, 1]
    f1.steering_command = current_steer / STEER_MAX  # normalize to [-1, 1]
    f1.throttle_command = torque

    # ══════════════════════════════════════════════════════════════════
    #  RERUN LOGGING
    # ══════════════════════════════════════════════════════════════════

    # ── 1. Agent raw outputs (before any processing) ──
    rr.log("agent/d_steer_raw", rr.Scalars(float(a_raw[0])))
    rr.log("agent/torque_raw", rr.Scalars(float(a_raw[1])))
    rr.log("agent/d_steer_clipped", rr.Scalars(d_steer))
    rr.log("agent/torque_clipped", rr.Scalars(float(a[1])))

    # ── 2. Steering pipeline: trace every step of the integration ──
    rr.log("steering/steer_vel_rad_s", rr.Scalars(sv))
    rr.log("steering/steer_before_integrate", rr.Scalars(steer_before))
    rr.log("steering/steer_after_integrate", rr.Scalars(current_steer))
    rr.log("steering/steer_delta_this_frame", rr.Scalars(current_steer - steer_before))
    rr.log("steering/nominal_dt", rr.Scalars(NOMINAL_DT))
    rr.log("steering/real_dt", rr.Scalars(real_dt))

    # ── 3. Final commands sent to AutoDRIVE ──
    rr.log("commands/steering_cmd", rr.Scalars(float(f1.steering_command)))
    rr.log("commands/throttle_cmd", rr.Scalars(float(f1.throttle_command)))

    # ── 4. Centerline features (are they reasonable?) ──
    rr.log("centerline/heading_diff_rad", rr.Scalars(dbg["heading_diff"]))
    rr.log("centerline/heading_diff_deg", rr.Scalars(np.degrees(dbg["heading_diff"])))
    rr.log("centerline/lateral_offset_m", rr.Scalars(dbg["lat_off"]))
    rr.log("centerline/nearest_wp_idx", rr.Scalars(dbg["nearest_wp_idx"]))

    # ── 5. Vehicle state from AutoDRIVE ──
    rr.log("vehicle/speed_m_s", rr.Scalars(dbg["vel"]))
    rr.log("vehicle/autodrive_steering_raw", rr.Scalars(dbg["autodrive_steering_raw"]))
    rr.log("vehicle/steer_rad", rr.Scalars(dbg["steer_rad"]))
    rr.log("vehicle/yaw_rate", rr.Scalars(dbg["yaw_rate"]))
    rr.log("vehicle/yaw_deg", rr.Scalars(np.degrees(dbg["theta"])))

    # ── 6. Map view: car position + trajectory + lookahead ──
    rr.log(
        "world/car",
        rr.Points2D([[dbg["x"], dbg["y"]]], colors=[[255, 60, 60]], radii=0.12),
    )
    # Car heading arrow
    arrow_len = 0.5
    rr.log(
        "world/car_heading",
        rr.Arrows2D(
            origins=[[dbg["x"], dbg["y"]]],
            vectors=[
                [
                    arrow_len * np.cos(dbg["theta"]),
                    arrow_len * np.sin(dbg["theta"]),
                ]
            ],
            colors=[[255, 60, 60]],
        ),
    )
    # Nearest centerline point
    ncx, ncy = _skeleton_pts[dbg["nearest_wp_idx"]]
    rr.log(
        "world/nearest_wp",
        rr.Points2D([[ncx, ncy]], colors=[[0, 255, 0]], radii=0.1),
    )
    # Lookahead waypoints (world frame)
    rr.log(
        "world/lookahead",
        rr.Points2D(dbg["lookahead_world"], colors=[[255, 200, 0]], radii=0.06),
    )
    rr.log(
        "world/lookahead_line",
        rr.LineStrips2D(
            [np.vstack([[dbg["x"], dbg["y"]], dbg["lookahead_world"]])],
            colors=[[255, 200, 0, 100]],
        ),
    )

    # ── 7. Car-frame view: LiDAR fan + lookahead ──
    lidar_ranges = dbg["lidar"]
    lidar_angles = np.linspace(-np.pi * 0.75, np.pi * 0.75, N_LIDAR_OBS)
    lidar_x = lidar_ranges * np.cos(lidar_angles)
    lidar_y = lidar_ranges * np.sin(lidar_angles)
    rr.log(
        "car_frame/lidar",
        rr.Points2D(
            np.stack([lidar_x, lidar_y], axis=-1),
            colors=[[0, 180, 255]],
            radii=0.02,
        ),
    )
    rr.log(
        "car_frame/lookahead",
        rr.Points2D(dbg["lookahead_local"], colors=[[255, 200, 0]], radii=0.08),
    )
    rr.log(
        "car_frame/origin",
        rr.Points2D([[0, 0]], colors=[[255, 60, 60]], radii=0.08),
    )

    # ── 8. Observation sanity checks ──
    # Log a few representative normalized lidar beams (front, left, right)
    front_idx = N_LIDAR_OBS // 2
    rr.log("obs/lidar_norm/front", rr.Scalars(float(obs_norm[front_idx])))
    rr.log("obs/lidar_norm/left", rr.Scalars(float(obs_norm[0])))
    rr.log("obs/lidar_norm/right", rr.Scalars(float(obs_norm[N_LIDAR_OBS - 1])))

    # Raw vs normalized for key features
    rr.log("obs/lidar_norm/raw_front_m", rr.Scalars(float(dbg["obs_raw"][front_idx])))

    # Snapshot of the full normalized obs as a bar chart (every 10th frame)
    if _frame_count % 10 == 0:
        rr.log("obs/full_vector", rr.BarChart(obs_norm))

    # ══════════════════════════════════════════════════════════════════

    try:
        sio.emit("Bridge", data=f1.generate_commands())
    except Exception as e:
        print(e)


if __name__ == "__main__":
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
