#!/usr/bin/env python3
"""Minimal RL racing agent. Uses nav2_amcl for localization.
Works on real car and sim with zero localization code.
No dependency on rustoracer — centerline is extracted via skeletonization."""

from pathlib import Path

import numpy as np
import rclpy
import tf2_ros
import torch
import torch.nn as nn
import yaml
from PIL import Image
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from scipy.ndimage import convolve, label, uniform_filter1d
from scipy.spatial import KDTree
from sensor_msgs.msg import Imu, LaserScan
from skimage.morphology import remove_small_objects, skeletonize
from std_msgs.msg import Float32
from tf_transformations import euler_from_quaternion

STEER_MAX = 0.5236
STEER_VEL_MAX = 3.2
SPEED_MAX = 22.88
N_LIDAR_RAW = 1080
N_LIDAR_OBS = 108
N_LOOK = 10
LIDAR_RANGE_MAX = 10.0
LIDAR_RANGE_MIN = 0.06

# ── Skeletonization helpers ─────────────────────────────────────────
_NBR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)


def _load_map_free_space(map_yaml_path: str) -> tuple[np.ndarray, float, list]:
    """Parse a ROS map YAML and return (free-space mask, resolution, origin)."""
    map_yaml = Path(map_yaml_path)
    with open(map_yaml) as f:
        meta = yaml.safe_load(f)

    img_path = (map_yaml.parent / meta["image"]).resolve()
    resolution = float(meta["resolution"])
    origin = meta["origin"]  # [x, y, θ]
    free_thresh = float(meta.get("free_thresh", 0.196))
    negate = int(meta.get("negate", 0))

    img = np.array(Image.open(str(img_path)).convert("L"), dtype=np.float64)
    occ = img / 255.0 if negate else (255.0 - img) / 255.0
    free = occ < free_thresh

    return free, resolution, origin, img.shape[0]


def _prune_branches(skel: np.ndarray, max_iters: int = 500) -> np.ndarray:
    """Iteratively remove endpoint pixels (nbr count == 1).
    A closed loop has no endpoints, so only dangling spurs are trimmed."""
    skel = skel.copy()
    for _ in range(max_iters):
        nbrs = convolve(skel.astype(np.int32), _NBR_KERNEL, mode="constant")
        endpoints = skel & (nbrs == 1)
        if not endpoints.any():
            break
        skel[endpoints] = False
    return skel


def _keep_largest_component(skel: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a binary image."""
    labeled, n = label(skel)
    if n <= 1:
        return skel
    sizes = np.bincount(labeled.ravel())[1:]  # skip background (0)
    largest = np.argmax(sizes) + 1
    return labeled == largest


def _order_points_nn(pts: np.ndarray) -> np.ndarray:
    """Order 2-D points into a continuous path via nearest-neighbour walk."""
    n = len(pts)
    tree = KDTree(pts)
    visited = np.zeros(n, dtype=bool)
    order = [0]
    visited[0] = True
    for _ in range(n - 1):
        _, nbr_idxs = tree.query(pts[order[-1]], k=min(64, n))
        for ni in nbr_idxs:
            if not visited[ni]:
                order.append(ni)
                visited[ni] = True
                break
    return pts[np.array(order)]


def _smooth_loop(pts: np.ndarray, win: int = 5) -> np.ndarray:
    """Circular (wrap-around) moving-average smooth for a closed path."""
    if len(pts) < 2 * win + 1:
        return pts
    pad = win
    padded_x = np.pad(pts[:, 0], pad, mode="wrap")
    padded_y = np.pad(pts[:, 1], pad, mode="wrap")
    sx = uniform_filter1d(padded_x, size=2 * pad + 1)[pad:-pad]
    sy = uniform_filter1d(padded_y, size=2 * pad + 1)[pad:-pad]
    return np.column_stack([sx, sy])


def load_centerline(map_yaml_path: str, smooth_win: int = 5) -> np.ndarray:
    """Extract a racing centerline (N×2, world coords) from a ROS map YAML.

    Pipeline:
      1. Threshold the map image into free-space.
      2. Skeletonize (topological thinning).
      3. Keep the largest connected component.
      4. Prune dangling branches so only the closed loop remains.
      5. Order pixels via nearest-neighbour walk.
      6. Convert pixel → world coordinates.
      7. Smooth with a circular moving average.
    """
    free, resolution, origin, img_h = _load_map_free_space(map_yaml_path)

    # Remove tiny free-space blobs that confuse skeletonization
    free = remove_small_objects(free, min_size=100)

    skel = skeletonize(free)
    skel = _keep_largest_component(skel)
    skel = _prune_branches(skel)

    ys, xs = np.where(skel)
    if len(ys) == 0:
        raise RuntimeError(
            f"Skeletonization of '{map_yaml_path}' produced no centerline pixels."
        )

    ordered = _order_points_nn(np.column_stack([xs, ys]).astype(np.float64))

    # Pixel → world  (image y-axis is flipped w.r.t. world y)
    wx = origin[0] + ordered[:, 0] * resolution
    wy = origin[1] + (img_h - 1 - ordered[:, 1]) * resolution
    centerline = np.column_stack([wx, wy])

    return _smooth_loop(centerline, win=smooth_win)


# ── RL agent ─────────────────────────────────────────────────────────
class Agent(nn.Module):
    def __init__(self, obs_dim, act=2, h=256):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, act),
        )

    def forward(self, x):
        return self.actor_mean(x)


class RacingAgentNode(Node):
    def __init__(self):
        super().__init__("racing_agent")

        self.declare_parameter("map_yaml", "maps/my_map.yaml")
        self.declare_parameter("checkpoint", "checkpoints/agent_final.pt")
        self.declare_parameter("throttle_scale", 1.0)

        map_yaml = self.get_parameter("map_yaml").value
        ckpt_path = self.get_parameter("checkpoint").value
        self.thr = self.get_parameter("throttle_scale").value

        # Load model
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        self.obs_mean = ckpt["obs_rms_mean"].numpy()
        self.obs_var = ckpt["obs_rms_var"].numpy()

        self.agent = Agent(self.obs_mean.shape[0])
        self.agent.load_state_dict(ckpt["model"], strict=False)
        self.agent.to(dev).eval()
        self.dev = dev

        # ── Centerline from skeletonization (replaces PySim) ─────────
        self.skel = load_centerline(map_yaml)
        self.n_wps = len(self.skel)
        seg = np.hypot(
            np.roll(self.skel[:, 0], -1) - self.skel[:, 0],
            np.roll(self.skel[:, 1], -1) - self.skel[:, 1],
        )
        self.look_step = max(1, round(1.0 / float(seg.mean())))
        self.lidar_idx = np.round(np.linspace(0, N_LIDAR_RAW - 1, N_LIDAR_OBS)).astype(
            int
        )

        # State
        self.x = self.y = self.theta = 0.0
        self.pose_ok = False
        self.speed = 0.0
        self.steer_fb = 0.0
        self.yaw_rate = 0.0
        self.lidar = np.full(N_LIDAR_RAW, LIDAR_RANGE_MAX)
        self.cur_steer = 0.0
        self.prev_t = None

        # TF: AMCL publishes map→world, bridge publishes world→roboracer_1
        self.tf_buf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buf, self)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # Subscribe to input topics only
        self.create_subscription(
            LaserScan, "/autodrive/roboracer_1/lidar", self._cb_lidar, qos
        )
        self.create_subscription(
            Float32, "/autodrive/roboracer_1/steering", self._cb_steer, qos
        )
        self.create_subscription(
            Float32, "/autodrive/roboracer_1/throttle", self._cb_thr, qos
        )
        self.create_subscription(Imu, "/autodrive/roboracer_1/imu", self._cb_imu, qos)

        self.pub_s = self.create_publisher(
            Float32, "/autodrive/roboracer_1/steering_command", qos
        )
        self.pub_t = self.create_publisher(
            Float32, "/autodrive/roboracer_1/throttle_command", qos
        )

        self.create_timer(0.025, self._loop)  # 40 Hz
        self.get_logger().info(
            f"Ready: wps={self.n_wps} step={self.look_step} thr={self.thr}"
        )

    # ── Callbacks ────────────────────────────────────────────────────
    def _cb_lidar(self, m):
        self.lidar = np.array(m.ranges, dtype=np.float64)

    def _cb_steer(self, m):
        self.steer_fb = float(m.data)

    def _cb_thr(self, m):
        pass

    def _cb_imu(self, m):
        self.yaw_rate = m.angular_velocity.z

    # ── Pose from AMCL TF tree ──────────────────────────────────────
    def _update_pose(self):
        try:
            tf = self.tf_buf.lookup_transform(
                "map",
                "roboracer_1",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.02),
            )
            self.x = tf.transform.translation.x
            self.y = tf.transform.translation.y
            q = tf.transform.rotation
            _, _, self.theta = euler_from_quaternion([q.x, q.y, q.z, q.w])
            self.pose_ok = True
        except Exception:
            pass

    # ── Build observation ────────────────────────────────────────────
    def _build_obs(self):
        lidar = np.clip(self.lidar[self.lidar_idx], LIDAR_RANGE_MIN, LIDAR_RANGE_MAX)
        x, y, th = self.x, self.y, self.theta
        ch, sh = np.cos(th), np.sin(th)

        wi = int(np.argmin(np.hypot(self.skel[:, 0] - x, self.skel[:, 1] - y)))
        ncx, ncy = self.skel[wi]
        nx, ny = self.skel[(wi + 1) % self.n_wps]
        cth = np.arctan2(ny - ncy, nx - ncx)
        hd = ((cth - th) + np.pi) % (2 * np.pi) - np.pi
        lat = -(x - ncx) * np.sin(cth) + (y - ncy) * np.cos(cth)

        lk = np.empty(N_LOOK * 2)
        for k in range(N_LOOK):
            wp = (wi + (k + 1) * self.look_step) % self.n_wps
            wx, wy = self.skel[wp]
            dx, dy = wx - x, wy - y
            lk[k * 2] = dx * ch + dy * sh
            lk[k * 2 + 1] = -dx * sh + dy * ch

        vel = np.clip(self.speed, -SPEED_MAX, SPEED_MAX)
        steer = self.steer_fb * STEER_MAX

        obs = np.concatenate([lidar, [vel, steer, self.yaw_rate, hd, lat], lk])
        return np.clip((obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8), -10, 10)

    # ── Control loop ─────────────────────────────────────────────────
    def _loop(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = np.clip(now - self.prev_t, 1e-4, 0.1) if self.prev_t else 0.025
        self.prev_t = now

        self._update_pose()
        if not self.pose_ok:
            return

        obs = self._build_obs()
        with torch.no_grad():
            act = self.agent(
                torch.tensor(obs, device=self.dev, dtype=torch.float32).unsqueeze(0)
            )
        a = act.cpu().numpy().flatten().clip(-1, 1)

        self.cur_steer = self.steer_fb * STEER_MAX
        self.cur_steer = np.clip(
            self.cur_steer + float(a[0]) * STEER_VEL_MAX * dt, -STEER_MAX, STEER_MAX
        )

        s = Float32()
        s.data = float(self.cur_steer / STEER_MAX)
        t = Float32()
        t.data = float(a[1]) * self.thr
        self.pub_s.publish(s)
        self.pub_t.publish(t)


def main(args=None):
    rclpy.init(args=args)
    node = RacingAgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
