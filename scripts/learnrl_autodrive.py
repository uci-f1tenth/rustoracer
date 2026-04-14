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

FIXES applied (relative to original):
  1. map_frame: world in slam_toolbox config — skeleton is now in world frame
  2. Steering integration uses real_dt (clamped) instead of fixed 1/60
  3. Steering state synced to actual AutoDRIVE feedback each frame
  4. Torque scaling removed (configurable via --throttle-scale)
  5. Speed clamped to physical limits (vehicle top speed 22.88 m/s)
  6. Frame-alignment diagnostics logged to Rerun
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
_parser.add_argument(
    "--throttle-scale",
    type=float,
    default=1.0,
    help="multiplier for throttle command (1.0 = full, 0.5 = half power)",
)
_cli = _parser.parse_args()

# ── Vehicle constants (from AutoDRIVE technical guide) ──
STEER_MAX = 0.5236  # rad (±30°)
STEER_VEL_MAX = 3.2  # rad/s (steering actuator rate)
SPEED_MAX = 22.88  # m/s (vehicle top speed per technical guide)
N_LIDAR_RAW = 1080  # AutoDRIVE LIDAR measurements per scan
N_LIDAR_OBS = 108  # downsampled beams to match sim training
N_LOOK = 10  # lookahead waypoints (must match training)

# ── Load checkpoint ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(_cli.checkpoint, map_location=DEVICE, weights_only=False)
obs_mean, obs_var = ckpt["obs_rms_mean"].numpy(), ckpt["obs_rms_var"].numpy()
OBS_DIM = obs_mean.shape[0]


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

# ── Centerline setup ──
_sim = PySim(_cli.map, 1, 1)
_skeleton_pts: np.ndarray = _sim.skeleton.reshape(-1, 2)
del _sim

N_WPS = len(_skeleton_pts)
_diffs = np.hypot(
    np.roll(_skeleton_pts[:, 0], -1) - _skeleton_pts[:, 0],
    np.roll(_skeleton_pts[:, 1], -1) - _skeleton_pts[:, 1],
)
LOOK_STEP = max(1, round(1.0 / float(_diffs.mean())))


# ── Helpers ──


def _yaw_from_quat(q: np.ndarray) -> float:
    """Extract yaw from [qx, qy, qz, qw] quaternion (AutoDRIVE convention)."""
    qx, qy, qz, qw = q
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


LIDAR_IDX = np.round(np.linspace(0, N_LIDAR_RAW - 1, N_LIDAR_OBS)).astype(int)

# Tracked steering angle [rad] — synced to AutoDRIVE feedback each frame
current_steer = 0.0


def get_obs(f):
    """Build the OBS_DIM-d observation vector from AutoDRIVE data."""

    # ── LiDAR (downsample 1080 → 108) ──
    lidar = f.lidar_range_array[LIDAR_IDX]

    # ── Vehicle states ──
    # Clamp speed to physical limits; protects against physics explosions
    vel = float(np.clip(f.speed, -SPEED_MAX, SPEED_MAX))
    steer_rad = f.steering * STEER_MAX  # sensor output [-1,1] → radians
    yaw_rate = f.angular_velocity[2]  # z-axis = yaw (z-up frame)

    # ── Car pose (IPS is in world frame; with map_frame:world they now match) ──
    x, y = f.position[0], f.position[1]
    theta = _yaw_from_quat(f.orientation_quaternion)
    ch, sh = np.cos(theta), np.sin(theta)

    # ── Nearest centerline waypoint ──
    wi = int(np.argmin(np.hypot(_skeleton_pts[:, 0] - x, _skeleton_pts[:, 1] - y)))

    # Distance to nearest waypoint (for diagnostics)
    ncx, ncy = _skeleton_pts[wi]
    dist_to_wp = np.hypot(x - ncx, y - ncy)

    # Centerline heading at wi
    nx, ny = _skeleton_pts[(wi + 1) % N_WPS]
    cth = np.arctan2(ny - ncy, nx - ncx)

    # Heading difference (wrapped to [-π, π])
    heading_diff = ((cth - theta) + np.pi) % (2.0 * np.pi) - np.pi

    # Signed lateral offset
    lat_off = -(x - ncx) * np.sin(cth) + (y - ncy) * np.cos(cth)

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
        dist_to_wp=dist_to_wp,
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
                    name="Frame Alignment",
                    origin="/alignment",
                ),
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

# ── Log static skeleton (once) ──
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

print(f"[bridge] OBS_DIM={OBS_DIM}, N_WPS={N_WPS}, LOOK_STEP={LOOK_STEP}")
print(f"[bridge] throttle_scale={_cli.throttle_scale}")
print(f"[bridge] obs_mean range: [{obs_mean.min():.3f}, {obs_mean.max():.3f}]")
print(f"[bridge] obs_var  range: [{obs_var.min():.6f}, {obs_var.max():.6f}]")

# ── Timing state ──
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

    # ── Measure real dt (clamped to avoid spikes) ──
    now = time.monotonic()
    if _prev_time is not None:
        real_dt = np.clip(now - _prev_time, 1e-4, 0.1)  # clamp to [0.1ms, 100ms]
    else:
        real_dt = 1.0 / 60.0  # first frame: assume 60 Hz
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

    d_steer = float(a[0])  # normalized steering velocity ∈ [-1, 1]
    throttle = float(a[1])  # normalized torque ∈ [-1, 1]

    # ── FIX 3: Sync steering state to actual AutoDRIVE feedback ──
    # This ensures we always integrate from the real steering angle,
    # matching how the Rust sim works (observation reflects actual state).
    current_steer = f1.steering * STEER_MAX

    # ── FIX 2: Integrate steering velocity using real_dt ──
    sv = d_steer * STEER_VEL_MAX  # rad/s
    steer_before = current_steer
    current_steer = np.clip(current_steer + sv * real_dt, -STEER_MAX, STEER_MAX)

    # ── Send commands to AutoDRIVE ──
    f1.steering_command = current_steer / STEER_MAX  # normalize to [-1, 1]
    f1.throttle_command = throttle * _cli.throttle_scale  # FIX 4: configurable scale

    # ══════════════════════════════════════════════════════════════════
    #  RERUN LOGGING
    # ══════════════════════════════════════════════════════════════════

    # ── 1. Agent raw outputs ──
    rr.log("agent/d_steer_raw", rr.Scalars(float(a_raw[0])))
    rr.log("agent/torque_raw", rr.Scalars(float(a_raw[1])))
    rr.log("agent/d_steer_clipped", rr.Scalars(d_steer))
    rr.log("agent/torque_clipped", rr.Scalars(throttle))

    # ── 2. Steering pipeline ──
    rr.log("steering/steer_vel_rad_s", rr.Scalars(sv))
    rr.log("steering/steer_before_integrate", rr.Scalars(steer_before))
    rr.log("steering/steer_after_integrate", rr.Scalars(current_steer))
    rr.log("steering/steer_delta_this_frame", rr.Scalars(current_steer - steer_before))
    rr.log("steering/real_dt", rr.Scalars(real_dt))

    # ── 3. Final commands sent to AutoDRIVE ──
    rr.log("commands/steering_cmd", rr.Scalars(float(f1.steering_command)))
    rr.log("commands/throttle_cmd", rr.Scalars(float(f1.throttle_command)))

    # ── 4. Centerline features ──
    rr.log("centerline/heading_diff_rad", rr.Scalars(dbg["heading_diff"]))
    rr.log("centerline/heading_diff_deg", rr.Scalars(np.degrees(dbg["heading_diff"])))
    rr.log("centerline/lateral_offset_m", rr.Scalars(dbg["lat_off"]))

    # ── 5. Frame alignment diagnostics (NEW) ──
    rr.log("alignment/dist_to_nearest_wp_m", rr.Scalars(dbg["dist_to_wp"]))
    rr.log("alignment/nearest_wp_idx", rr.Scalars(dbg["nearest_wp_idx"]))
    rr.log("alignment/car_x", rr.Scalars(dbg["x"]))
    rr.log("alignment/car_y", rr.Scalars(dbg["y"]))
    rr.log("alignment/yaw_deg", rr.Scalars(np.degrees(dbg["theta"])))
    rr.log(
        "alignment/centerline_heading_deg",
        rr.Scalars(np.degrees(dbg["centerline_heading"])),
    )

    # ── 6. Vehicle state ──
    rr.log("vehicle/speed_m_s", rr.Scalars(dbg["vel"]))
    rr.log("vehicle/autodrive_steering_raw", rr.Scalars(dbg["autodrive_steering_raw"]))
    rr.log("vehicle/steer_rad", rr.Scalars(dbg["steer_rad"]))
    rr.log("vehicle/yaw_rate", rr.Scalars(dbg["yaw_rate"]))

    # ── 7. Map view ──
    rr.log(
        "world/car",
        rr.Points2D([[dbg["x"], dbg["y"]]], colors=[[255, 60, 60]], radii=0.12),
    )
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
    ncx, ncy = _skeleton_pts[dbg["nearest_wp_idx"]]
    rr.log(
        "world/nearest_wp",
        rr.Points2D([[ncx, ncy]], colors=[[0, 255, 0]], radii=0.1),
    )
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

    # ── 8. Car-frame view ──
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

    # ── 9. Observation checks ──
    front_idx = N_LIDAR_OBS // 2
    rr.log("obs/lidar_norm/front", rr.Scalars(float(obs_norm[front_idx])))
    rr.log("obs/lidar_norm/left", rr.Scalars(float(obs_norm[0])))
    rr.log("obs/lidar_norm/right", rr.Scalars(float(obs_norm[N_LIDAR_OBS - 1])))
    rr.log("obs/lidar_norm/raw_front_m", rr.Scalars(float(dbg["obs_raw"][front_idx])))

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
