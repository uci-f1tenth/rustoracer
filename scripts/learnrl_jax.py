"""learnrl_jax.py — F1tenth RL in pure JAX.

Why this is faster than the Rust + PyTorch stack:

  • lax.scan compiles the entire T-step rollout into one XLA program.
    Zero Python overhead per environment step, zero CPU↔GPU copies.
  • jax.vmap parallelises all N environments natively on the GPU (replaces rayon).
  • f32 throughout — typically 2× faster than f64 on GPUs.
  • jit-compiled PPO update fuses every matmul and gradient step into a
    single device program.
  • No PyO3 boundary, no Gymnasium wrapper, no Torch tensor boxing.

Usage (CPU):
  uv run --with "jax[cpu]" --with optax --with scipy --with scikit-image \\
         --with pillow --with pyyaml --with tqdm --with tyro \\
         scripts/learnrl_jax.py

Usage (GPU/CUDA):
  uv run --with "jax[cuda12]" --with optax --with scipy --with scikit-image \\
         --with pillow --with pyyaml --with tqdm --with tyro \\
         scripts/learnrl_jax.py
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import optax
import scipy.ndimage as ndi
import scipy.signal
import skimage.morphology
import tyro
import tqdm
import yaml as pyyaml
from PIL import Image
from scipy.spatial import KDTree

# ── silence skimage deprecation warnings on import ────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")


# ── CLI config ─────────────────────────────────────────────────────────────────


@dataclass
class Args:
    yaml: str = "maps/my_map.yaml"
    """Path to the map YAML file."""
    num_envs: int = 1024
    """Number of parallel environments."""
    num_steps: int = 128
    """Rollout length per policy update."""
    total_timesteps: int = 200_000_000
    """Total training timesteps."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256
    seed: int = 42
    max_steps: int = 10_000
    dr_frac: float = 0.2
    """Domain-randomisation fraction (±fraction of default param values)."""
    save_interval: int = 50
    save_dir: str = "checkpoints"


# ── Physics constants (f32) ────────────────────────────────────────────────────

G = np.float32(9.81)
STEER_MIN = np.float32(-0.5236)
STEER_MAX = np.float32(0.5236)
STEER_VEL_MAX = np.float32(3.2)
V_MIN = np.float32(-5.0)
V_MAX = np.float32(20.0)
WIDTH = np.float32(0.27)
LENGTH = np.float32(0.50)
CD_A_RHO_HALF = np.float32(0.5 * 0.3 * 0.04 * 1.225)
V_S = np.float32(0.2)
V_B = np.float32(0.05)
V_MIN_DYN = np.float32(0.1)
OMEGA_TAU = np.float32(0.02)
PI = np.float32(np.pi)

N_BEAMS = 108
FOV = np.float32(270.0 * np.pi / 180.0)
MIN_RANGE = np.float32(0.06)
MAX_RANGE = np.float32(10.0)
N_LOOK = 10

# Default car parameters
_P_DICT = dict(
    lf=0.15532, lr=0.16868, h=0.01434, mass=3.906, i_z=0.04712,
    v_switch=7.319, a_max=9.51, r_w=0.059, i_yw=0.008, t_sb=0.5, t_se=0.5,
    p_cx1=1.6411, p_dx1=1.1739, p_ex1=0.464,  p_kx1=5.42,  p_hx1=1.2297e-3,
    p_vx1=-8.8098e-6, r_bx1=13.276, r_bx2=-13.778, r_cx1=1.2568, r_ex1=0.6522,
    r_hx1=5.0722e-3, p_cy1=1.3507, p_dy1=1.0489, p_ey1=-7.4722e-3, p_ky1=-5.33,
    r_by1=7.1433,  r_by2=9.1916,  r_by3=-2.7856e-2, r_cy1=1.0719, r_ey1=-0.2757,
    r_hy1=5.7448e-6, r_vy1=-2.7825e-2, r_vy4=12.12, r_vy5=1.9, r_vy6=-10.704,
)
P_DEF: dict[str, np.float32] = {k: np.float32(v) for k, v in _P_DICT.items()}

# Params that are randomised per episode
RAND_KEYS = ("lf", "lr", "h", "mass", "i_z", "v_switch", "a_max", "r_w", "i_yw",
             "p_dx1", "p_dy1", "p_kx1", "p_ky1")


# ── MAP LOADING (Python / scipy / skimage, runs once at startup) ───────────────


class MapData:
    """Holds map arrays as JAX tensors for use inside JIT-compiled code."""

    def __init__(self, yaml_path: str,
                 edt_thresh_frac: float = 0.1,
                 sg_window: int = 11, sg_degree: int = 3):
        meta = pyyaml.safe_load(open(yaml_path))
        img_path = Path(yaml_path).parent / meta["image"]
        img = np.array(Image.open(img_path).convert("L"))    # (H, W) uint8
        res = float(meta["resolution"])
        ox = float(meta["origin"][0])
        oy = float(meta["origin"][1])
        otheta = float(meta["origin"][2]) if len(meta["origin"]) > 2 else 0.0
        h, w = img.shape

        obstacle = img < 210    # bool (H, W), True = wall
        free = ~obstacle

        # EDT: distance from each free pixel to nearest wall, in metres
        edt_np = ndi.distance_transform_edt(free).astype(np.float32) * res

        # Center-line skeleton: skeletonise the "fat" interior of the free
        # corridor (pixels whose EDT exceeds edt_thresh_frac of the maximum),
        # then prune stub endpoints to expose the main loop.
        edt_px = ndi.distance_transform_edt(free)    # pixels, for thresholding
        thresh = float(edt_px.max() * edt_thresh_frac)
        wide_free = free & (edt_px >= thresh)
        thin = skimage.morphology.skeletonize(wide_free)
        ordered = _extract_loop(thin, ox, oy, res, otheta, h)
        assert len(ordered) >= 2, "Could not extract skeleton loop from map"
        smoothed = np.array(
            scipy.signal.savgol_filter(
                np.asarray(ordered, dtype=np.float64),
                sg_window, sg_degree, axis=0, mode="wrap"),
            dtype=np.float32)    # (S, 2)

        # LUT: for each pixel, index of nearest skeleton point
        tree = KDTree(smoothed)
        ys_g, xs_g = np.mgrid[0:h, 0:w]
        world_x = xs_g.ravel().astype(np.float32) * res + ox
        world_y = (h - 1 - ys_g.ravel()).astype(np.float32) * res + oy
        _, lut = tree.query(np.c_[world_x, world_y])

        # Beam (sin, cos) pairs, precomputed
        angles = np.linspace(-FOV / 2, FOV / 2, N_BEAMS, dtype=np.float32)
        beam_sc = np.stack([np.sin(angles), np.cos(angles)], axis=1)  # (N_BEAMS,2)

        # look_step: number of skeleton points ≈ 1 m
        seg = np.linalg.norm(smoothed[1:] - smoothed[:-1], axis=1)
        look_step = max(1, round(1.0 / seg.mean()))

        self.edt = jnp.array(edt_np)                             # (H, W) f32
        self.skeleton = jnp.array(smoothed)                      # (S, 2) f32
        self.skel_lut = jnp.array(lut.astype(np.int32))          # (H*W,) i32
        self.beam_sc = jnp.array(beam_sc)                        # (N_BEAMS, 2) f32
        self.w = w
        self.h = h
        self.res = np.float32(res)
        self.inv_res = np.float32(1.0 / res)
        self.ox = np.float32(ox)
        self.oy = np.float32(oy)
        self.n_wps = int(len(smoothed))
        self.look_step = int(look_step)
        self.obs_dim = N_BEAMS + 3 + 2 + N_LOOK * 2


def _nbrs8(p: tuple, pts: set) -> list:
    x, y = p
    return [(x + dx, y + dy)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if (dx, dy) != (0, 0) and (x + dx, y + dy) in pts]


def _largest_component(pts: set) -> set:
    visited: set = set()
    best: list = []
    for p in pts:
        if p in visited:
            continue
        comp: list = []
        q = [p]
        visited.add(p)
        for cur in q:
            comp.append(cur)
            for n in _nbrs8(cur, pts):
                if n not in visited:
                    visited.add(n)
                    q.append(n)
        if len(comp) > len(best):
            best = comp
    return set(best)


def _extract_loop(thin_img: np.ndarray, ox, oy, res, otheta, h) -> list:
    """Trace the main skeleton loop.

    Strategy:
      1. Take the largest 8-connected component.
      2. Iteratively prune degree-1 endpoints (removes stub branches).
      3. BFS from one neighbour of the start pixel to the other.
    """
    ys, xs = np.where(thin_img)
    pts = set(zip(xs.tolist(), ys.tolist()))
    if not pts:
        raise RuntimeError("Empty skeleton")

    pts = _largest_component(pts)

    # Prune stub branches: remove endpoints iteratively
    for _ in range(500):
        endpoints = [p for p in pts if len(_nbrs8(p, pts)) < 2]
        if not endpoints:
            break
        pts -= set(endpoints)
    if not pts:
        raise RuntimeError("Skeleton collapsed to empty after pruning")

    # Origin pixel → start node nearest to map origin
    origin_px = (-ox / res, (h - 1) + oy / res)
    start = min(pts, key=lambda p: (p[0] - origin_px[0]) ** 2
                + (p[1] - origin_px[1]) ** 2)

    adj = _nbrs8(start, pts)
    if len(adj) < 2:
        raise RuntimeError("Start node has fewer than 2 neighbours")

    src, targets = adj[0], set(adj[1:])
    parent: dict = {src: src}
    queue = [src]
    found = None
    for cur in queue:
        for n in _nbrs8(cur, pts):
            if n == start or n in parent:
                continue
            parent[n] = cur
            if n in targets:
                found = n
                break
            queue.append(n)
        if found:
            break

    path = [start]
    if found:
        p = found
        while p != src:
            path.append(p)
            p = parent[p]
        path.append(src)
    path.reverse()

    world = [[px * res + ox, (h - 1 - py) * res + oy] for px, py in path]
    if len(world) >= 2:
        dx0, dy0 = world[1][0] - world[0][0], world[1][1] - world[0][1]
        err = ((np.arctan2(dy0, dx0) - otheta + 3 * np.pi) % (2 * np.pi)) - np.pi
        if abs(err) > np.pi / 2:
            world[1:] = world[1:][::-1]
    return world


# ── JAX car dynamics ───────────────────────────────────────────────────────────
# Car dynamic state: (9,) f32 = [x, y, delta, v, psi, psi_dot, beta, ωf, ωr]
# Car params: dict of f32 scalars (vmapped over N envs simultaneously).


def _tire_fx_pure(s, f_z, p):
    f_z = jnp.maximum(f_z, 0.)
    kx = -s + p["p_hx1"]
    d_x = p["p_dx1"] * f_z
    b_x = jnp.where(jnp.abs(d_x) > 1e-10,
                    f_z * p["p_kx1"] / (p["p_cx1"] * d_x), 0.)
    bkx = b_x * kx
    return (d_x * jnp.sin(p["p_cx1"] * jnp.arctan(
        bkx - p["p_ex1"] * (bkx - jnp.arctan(bkx)))) + f_z * p["p_vx1"])


def _tire_fy_pure(alpha, f_z, p):
    f_z = jnp.maximum(f_z, 0.)
    d_y = p["p_dy1"] * f_z
    b_y = jnp.where(jnp.abs(d_y) > 1e-10,
                    f_z * p["p_ky1"] / (p["p_cy1"] * d_y), 0.)
    ba = b_y * alpha
    fy = d_y * jnp.sin(p["p_cy1"] * jnp.arctan(
        ba - p["p_ey1"] * (ba - jnp.arctan(ba))))
    return fy, p["p_dy1"]


def _tire_fx_combined(s, alpha, fx0, p):
    kappa = -s
    b_xa = p["r_bx1"] * jnp.cos(jnp.arctan(p["r_bx2"] * kappa))
    bs = b_xa * p["r_hx1"]
    denom = jnp.cos(p["r_cx1"] * jnp.arctan(bs - p["r_ex1"] * (bs - jnp.arctan(bs))))
    d_xa = jnp.where(jnp.abs(denom) > 1e-10, fx0 / denom, fx0)
    ba = b_xa * (alpha + p["r_hx1"])
    return d_xa * jnp.cos(p["r_cx1"] * jnp.arctan(ba - p["r_ex1"] * (ba - jnp.arctan(ba))))


def _tire_fy_combined(s, alpha, mu_y, f_z, fy0, p):
    kappa = -s
    kappa_s = kappa + p["r_hy1"]
    b_yk = p["r_by1"] * jnp.cos(jnp.arctan(p["r_by2"] * (alpha - p["r_by3"])))
    bsh = b_yk * p["r_hy1"]
    denom = jnp.cos(p["r_cy1"] * jnp.arctan(bsh - p["r_ey1"] * (bsh - jnp.arctan(bsh))))
    d_yk = jnp.where(jnp.abs(denom) > 1e-10, fy0 / denom, fy0)
    d_vyk = mu_y * f_z * p["r_vy1"] * jnp.cos(jnp.arctan(p["r_vy4"] * alpha))
    s_vyk = d_vyk * jnp.sin(p["r_vy5"] * jnp.arctan(p["r_vy6"] * kappa))
    bk = b_yk * kappa_s
    return (d_yk * jnp.cos(p["r_cy1"] * jnp.arctan(bk - p["r_ey1"] * (bk - jnp.arctan(bk))))
            + s_vyk)


def _std_dynamics(s, torque, sv, p):
    """Continuous-time derivatives of the 9-state single-track model."""
    x, y, delta, v, psi, psi_dot, beta, omega_f, omega_r = s
    omega_f = jnp.maximum(omega_f, 0.)
    omega_r = jnp.maximum(omega_r, 0.)
    lwb = p["lf"] + p["lr"]

    safe_v = v > V_MIN_DYN
    vc = v * jnp.cos(beta)
    vs = v * jnp.sin(beta)
    vc_safe = jnp.where(safe_v, vc, jnp.float32(1.))

    alpha_f = jnp.where(safe_v,
                        jnp.arctan((vs + psi_dot * p["lf"]) / vc_safe) - delta, 0.)
    alpha_r = jnp.where(safe_v,
                        jnp.arctan((vs - psi_dot * p["lr"]) / vc_safe), 0.)

    a_est = torque / (p["mass"] * p["r_w"])
    f_zf = p["mass"] * (-a_est * p["h"] + G * p["lr"]) / lwb
    f_zr = p["mass"] * (a_est * p["h"] + G * p["lf"]) / lwb

    u_wf = jnp.maximum(
        v * jnp.cos(beta) * jnp.cos(delta)
        + (v * jnp.sin(beta) + p["lf"] * psi_dot) * jnp.sin(delta), 0.)
    u_wr = jnp.maximum(v * jnp.cos(beta), 0.)

    s_f = jnp.clip(1. - p["r_w"] * omega_f / jnp.maximum(u_wf, V_MIN_DYN), -1., 1.)
    s_r = jnp.clip(1. - p["r_w"] * omega_r / jnp.maximum(u_wr, V_MIN_DYN), -1., 1.)

    fx0_f = _tire_fx_pure(s_f, f_zf, p)
    fx0_r = _tire_fx_pure(s_r, f_zr, p)
    fy0_f, mu_yf = _tire_fy_pure(alpha_f, f_zf, p)
    fy0_r, mu_yr = _tire_fy_pure(alpha_r, f_zr, p)
    f_xf = _tire_fx_combined(s_f, alpha_f, fx0_f, p)
    f_xr = _tire_fx_combined(s_r, alpha_r, fx0_r, p)
    f_yf = _tire_fy_combined(s_f, alpha_f, mu_yf, f_zf, fy0_f, p)
    f_yr = _tire_fy_combined(s_r, alpha_r, mu_yr, f_zr, fy0_r, p)

    t_b = jnp.where(torque > 0., 0., torque)
    t_e = jnp.where(torque > 0., torque, 0.)
    f_drag = CD_A_RHO_HALF * v * jnp.abs(v)

    sin_db, cos_db = jnp.sin(delta - beta), jnp.cos(delta - beta)
    d_v_dyn = (1. / p["mass"]) * (
        -f_yf * sin_db + f_yr * jnp.sin(beta)
        + f_xr * jnp.cos(beta) + f_xf * cos_db - f_drag)
    dd_psi_dyn = (1. / p["i_z"]) * (
        f_yf * jnp.cos(delta) * p["lf"] - f_yr * p["lr"]
        + f_xf * jnp.sin(delta) * p["lf"])
    d_beta_dyn = jnp.where(safe_v,
                           -psi_dot + (1. / (p["mass"] * v)) * (
                               f_yf * cos_db + f_yr * jnp.cos(beta)
                               - f_xr * jnp.sin(beta) + f_xf * sin_db), 0.)

    def wheel_dyn(omega, f_x, tb_frac, te_frac):
        net = -p["r_w"] * f_x + tb_frac * t_b + te_frac * t_e
        return jnp.where(omega > 0., net / p["i_yw"],
                         jnp.maximum(net, 0.) / p["i_yw"])

    d_omf_dyn = wheel_dyn(omega_f, f_xf, p["t_sb"], p["t_se"])
    d_omr_dyn = wheel_dyn(omega_r, f_xr, 1. - p["t_sb"], 1. - p["t_se"])

    # Kinematic model (blended in at low speed)
    d_v_ks = a_est - f_drag / p["mass"]
    cos2_d = jnp.cos(delta) ** 2
    tan_d = jnp.tan(delta)
    tan_lr_lwb = tan_d * p["lr"] / lwb
    safe_cos2 = jnp.abs(cos2_d) > 1e-10
    d_beta_ks = jnp.where(safe_cos2,
                           p["lr"] * sv / (lwb * cos2_d * (1. + tan_lr_lwb ** 2)), 0.)
    dd_psi_ks = jnp.where(safe_cos2,
                           (1. / lwb) * (
                               a_est * jnp.cos(beta) * tan_d
                               - v * jnp.sin(beta) * d_beta_ks * tan_d
                               + v * jnp.cos(beta) / cos2_d * sv), 0.)
    d_psi_ks = v * jnp.cos(beta) / lwb * tan_d
    d_omf_ks = (u_wf / p["r_w"] - omega_f) / OMEGA_TAU
    d_omr_ks = (u_wr / p["r_w"] - omega_r) / OMEGA_TAU

    w_dyn = 0.5 * jnp.tanh((v - V_S) / V_B) + 0.5
    w_ks = 1. - w_dyn

    return jnp.array([
        v * jnp.cos(beta + psi),
        v * jnp.sin(beta + psi),
        sv,
        w_dyn * d_v_dyn  + w_ks * d_v_ks,
        w_dyn * psi_dot  + w_ks * d_psi_ks,
        w_dyn * dd_psi_dyn + w_ks * dd_psi_ks,
        w_dyn * d_beta_dyn + w_ks * d_beta_ks,
        w_dyn * d_omf_dyn  + w_ks * d_omf_ks,
        w_dyn * d_omr_dyn  + w_ks * d_omr_ks,
    ])


def _rk4(s, dt, f):
    k1 = f(s)
    k2 = f(s + k1 * (dt / 2))
    k3 = f(s + k2 * (dt / 2))
    k4 = f(s + k3 * dt)
    return s + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)


def _steering_constraint(steer, sv):
    sv = jnp.clip(sv, -STEER_VEL_MAX, STEER_VEL_MAX)
    sv = jnp.where((sv < 0.) & (steer <= STEER_MIN), 0., sv)
    sv = jnp.where((sv > 0.) & (steer >= STEER_MAX), 0., sv)
    return sv


# ── JAX environment functions (closed over a fixed MapData) ───────────────────


def make_sim_fns(map_data: MapData, max_steps: int, dr_frac: float,
                 dt: float = 1. / 60., substeps: int = 6):
    """Return JIT-compiled batched environment functions.

    Returns:
        batched_step : (state, actions) → (new_state, (reward, term, trunc))
        batched_obs  : state → (N, obs_dim)
        batched_init : rng_keys(N,2) → state
    """
    edt = map_data.edt
    skeleton = map_data.skeleton
    skel_lut = map_data.skel_lut
    beam_sc = map_data.beam_sc
    W, H = map_data.w, map_data.h
    res, inv_res = map_data.res, map_data.inv_res
    ox, oy = map_data.ox, map_data.oy
    n_wps = jnp.int32(map_data.n_wps)
    look_step = jnp.int32(map_data.look_step)
    sub_dt = jnp.float32(dt / substeps)
    Wf, Hf = jnp.float32(W), jnp.float32(H)

    def _pos_to_px(x, y):
        px = jnp.int32(jnp.clip((x - ox) * inv_res, 0., Wf - 1.))
        py = jnp.int32(jnp.clip((Hf - 1.) - (y - oy) * inv_res, 0., Hf - 1.))
        return px, py

    def _collides(dyn):
        x, y, _, _, theta, _, _, _, _ = dyn
        not_finite = ~(jnp.isfinite(x) & jnp.isfinite(y) & jnp.isfinite(theta))
        px_f = (x - ox) * inv_res
        py_f = (Hf - 1.) - (y - oy) * inv_res
        in_map = (px_f >= 0.) & (px_f < Wf) & (py_f >= 0.) & (py_f < Hf)
        pxi = jnp.int32(jnp.clip(px_f, 0., Wf - 1.))
        pyi = jnp.int32(jnp.clip(py_f, 0., Hf - 1.))
        c = jnp.where(in_map, edt[pyi, pxi], 0.)
        R_SQ = jnp.float32((LENGTH / 2.) ** 2 + (WIDTH / 2.) ** 2)
        no_col = c * c > R_SQ
        # Check 8 car rectangle sample points in pixel space
        sa, ca = jnp.sin(theta), jnp.cos(theta)
        hl = jnp.float32(LENGTH / 2. * inv_res)
        hw = jnp.float32(WIDTH / 2. * inv_res)
        offs = jnp.array([[ hl,  hw], [ hl, -hw], [-hl,  hw], [-hl, -hw],
                          [ hl, 0.],  [-hl, 0.],  [0.,   hw], [0.,  -hw]])
        dx = offs[:, 0] * ca - offs[:, 1] * sa
        dy = offs[:, 0] * sa + offs[:, 1] * ca
        cxs = jnp.int32(jnp.clip(px_f + dx, 0., Wf - 1.))
        cys = jnp.int32(jnp.clip(py_f + dy, 0., Hf - 1.))
        any_col = jnp.any(edt[cys, cxs] < res)
        return not_finite | jnp.where(no_col, jnp.bool_(False), any_col)

    def _raycast(lx, ly, bdx, bdy):
        """EDT sphere-tracing for one ray via lax.while_loop."""
        px0 = (lx - ox) * inv_res
        py0 = (Hf - 1.) - (ly - oy) * inv_res
        dpx, dpy = bdx * inv_res, -bdy * inv_res

        def cond(carry):
            t, _result, done = carry
            return ~done & (t < MAX_RANGE)

        def body(carry):
            t, result, done = carry
            pxf = px0 + t * dpx
            pyf = py0 + t * dpy
            ib = (pxf >= 0.) & (pxf < Wf) & (pyf >= 0.) & (pyf < Hf)
            pxi = jnp.int32(jnp.clip(pxf, 0., Wf - 1.))
            pyi = jnp.int32(jnp.clip(pyf, 0., Hf - 1.))
            d = jnp.where(ib, edt[pyi, pxi], 0.)
            hit = ~ib | (d < res * 0.5)
            new_t = t + jnp.maximum(d, res * 0.5)
            new_result = jnp.where(hit, t, result)
            return new_t, new_result, done | hit

        _, result, _ = lax.while_loop(
            cond, body, (MIN_RANGE, jnp.float32(MAX_RANGE), jnp.bool_(False)))
        return result

    def _compute_obs(dyn, wp_idx, p):
        x, y, delta, v, psi, psi_dot, *_ = dyn
        sh, ch = jnp.sin(psi), jnp.cos(psi)
        lx, ly = x + p["lf"] * ch, y + p["lf"] * sh

        def cast(sc):
            sa, ca = sc
            return _raycast(lx, ly, ch * ca - sh * sa, sh * ca + ch * sa)

        scans = jax.vmap(cast)(beam_sc)    # (N_BEAMS,)

        wp_pt = skeleton[wp_idx]
        wp_next = skeleton[(wp_idx + 1) % n_wps]
        cth = jnp.arctan2(wp_next[1] - wp_pt[1], wp_next[0] - wp_pt[0])
        heading_err = ((cth - psi) + PI) % (2. * PI) - PI
        lat_err = -(x - wp_pt[0]) * jnp.sin(cth) + (y - wp_pt[1]) * jnp.cos(cth)

        look_idx = (wp_idx + jnp.arange(1, N_LOOK + 1, dtype=jnp.int32) * look_step) % n_wps
        look_pts = skeleton[look_idx]    # (N_LOOK, 2)
        dx, dy = look_pts[:, 0] - x, look_pts[:, 1] - y
        look_local = jnp.stack([dx * ch + dy * sh, -dx * sh + dy * ch], axis=1).ravel()

        return jnp.concatenate([scans, jnp.array([v, delta, psi_dot]),
                                 jnp.array([heading_err, lat_err]), look_local])

    def _random_state(rng):
        rng, k1, k2 = jax.random.split(rng, 3)
        ri = jax.random.randint(k1, (), 0, n_wps)
        pt = skeleton[ri]
        pt_next = skeleton[(ri + 1) % n_wps]
        th = jnp.arctan2(pt_next[1] - pt[1], pt_next[0] - pt[0])

        pkeys = jax.random.split(k2, len(RAND_KEYS))
        p = dict(P_DEF)
        for i, k in enumerate(RAND_KEYS):
            scale = 1. + jnp.float32(dr_frac) * (jax.random.uniform(pkeys[i]) * 2. - 1.)
            p[k] = P_DEF[k] * scale

        dyn = jnp.array([pt[0], pt[1], 0., 0., th, 0., 0., 0., 0.])
        return dyn, ri, p, rng

    def _single_step(state, action):
        dyn, wp_idx, steps, rng, p = (
            state["dyn"], state["wp_idx"], state["steps"],
            state["rng"], state["p"])

        d_steer = jnp.clip(action[0], -1., 1.)
        torque_n = jnp.clip(action[1], -1., 1.)
        sv = _steering_constraint(dyn[2], d_steer * STEER_VEL_MAX)
        torque = torque_n * p["mass"] * p["a_max"] * p["r_w"]

        def substep(d, _):
            return _rk4(d, sub_dt, lambda s: _std_dynamics(s, torque, sv, p)), None

        dyn_new, _ = lax.scan(substep, dyn, None, length=substeps)

        x, y, delta, v, psi, psi_dot, beta, omf, omr = dyn_new
        delta = jnp.clip(delta, STEER_MIN, STEER_MAX)
        v = jnp.clip(v, V_MIN, V_MAX)
        psi = (psi + PI) % (2. * PI) - PI
        dyn_new = jnp.array([x, y, delta, v, psi, psi_dot, beta,
                              jnp.maximum(omf, 0.), jnp.maximum(omr, 0.)])

        # Reward
        px_new, py_new = _pos_to_px(x, y)
        new_wi = skel_lut[py_new * W + px_new]
        d_wp = jnp.float32(new_wi - wp_idx)
        d_wp = jnp.where(d_wp >  jnp.float32(n_wps) / 2., d_wp - jnp.float32(n_wps), d_wp)
        d_wp = jnp.where(d_wp < -jnp.float32(n_wps) / 2., d_wp + jnp.float32(n_wps), d_wp)

        edt_val = edt[py_new, px_new]
        terminated = _collides(dyn_new)
        truncated = (steps + jnp.int32(1)) >= jnp.int32(max_steps)
        reward = (d_wp / jnp.float32(n_wps) * 100. * (1. + jnp.maximum(v, 0.) / 10.)
                  - 0.1 * jnp.exp(-3. * edt_val)
                  - jnp.where(terminated, 100., 0.))

        # Autoreset (always compute; select via jnp.where for XLA friendliness)
        reset_dyn, reset_ri, reset_p, new_rng = _random_state(rng)
        done = terminated | truncated
        final_dyn = jnp.where(done, reset_dyn, dyn_new)
        final_wi = jnp.where(done, reset_ri, new_wi)
        final_steps = jnp.where(done, jnp.int32(0), steps + jnp.int32(1))
        final_p = jax.tree.map(lambda a, b: jnp.where(done, a, b), reset_p, p)

        new_state = {"dyn": final_dyn, "wp_idx": final_wi,
                     "steps": final_steps, "rng": new_rng, "p": final_p}
        return new_state, (reward, terminated, truncated)

    def _single_obs(state):
        return _compute_obs(state["dyn"], state["wp_idx"], state["p"])

    def _init_single(rng):
        dyn, ri, p, rng = _random_state(rng)
        return {"dyn": dyn, "wp_idx": ri, "steps": jnp.int32(0), "rng": rng, "p": p}

    batched_step = jax.jit(jax.vmap(_single_step))
    batched_obs  = jax.jit(jax.vmap(_single_obs))
    batched_init = jax.jit(jax.vmap(_init_single))
    return batched_step, batched_obs, batched_init


# ── Policy (hand-rolled MLP, no extra deps beyond Optax) ─────────────────────

LOG_STD_MIN, LOG_STD_MAX = jnp.float32(-5.), jnp.float32(0.5)


def init_policy(key, obs_dim: int, hidden: int, act_dim: int = 2):
    def layer(k, i, o, std=float(np.sqrt(2))):
        return {"w": jax.random.normal(k, (i, o)) * float(std / np.sqrt(i)),
                "b": jnp.zeros(o)}

    ks = jax.random.split(key, 6)
    return {
        "actor": [layer(ks[0], obs_dim, hidden),
                  layer(ks[1], hidden,  hidden),
                  layer(ks[2], hidden,  act_dim, std=0.01)],
        "actor_logstd": jnp.full((act_dim,), -0.5, dtype=jnp.float32),
        "critic": [layer(ks[3], obs_dim, hidden),
                   layer(ks[4], hidden,  hidden),
                   layer(ks[5], hidden,  1, std=1.)],
    }


def _mlp(layers, x):
    for lay in layers[:-1]:
        x = jax.nn.tanh(x @ lay["w"] + lay["b"])
    return x @ layers[-1]["w"] + layers[-1]["b"]


def _policy_forward(params, obs, key):
    """Sample action; return (action, log_prob, value)."""
    mean = _mlp(params["actor"], obs)
    log_std = jnp.clip(params["actor_logstd"], LOG_STD_MIN, LOG_STD_MAX)
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, mean.shape)
    action = mean + std * noise
    lp = (-0.5 * ((action - mean) / std) ** 2
          - jnp.log(std) - 0.5 * jnp.log(2. * jnp.pi)).sum(-1)
    value = _mlp(params["critic"], obs).squeeze(-1)
    return action, lp, value


def _policy_evaluate(params, obs, action):
    mean = _mlp(params["actor"], obs)
    log_std = jnp.clip(params["actor_logstd"], LOG_STD_MIN, LOG_STD_MAX)
    std = jnp.exp(log_std)
    lp = (-0.5 * ((action - mean) / std) ** 2
          - jnp.log(std) - 0.5 * jnp.log(2. * jnp.pi)).sum(-1)
    entropy = (jnp.log(std) + 0.5 * jnp.log(2. * jnp.pi * jnp.e)).sum(-1)
    value = _mlp(params["critic"], obs).squeeze(-1)
    return lp, entropy, value


# ── PPO rollout + update (fully JIT-compiled) ─────────────────────────────────


def make_ppo_fns(batched_step, batched_obs, optimizer, args: Args):
    """Build JIT-compiled rollout and PPO update functions.

    Both functions are closed over `optimizer` which is a pure optax chain.
    Changing the optimizer object requires rebuilding these functions.
    """
    N   = args.num_envs
    T   = args.num_steps
    B   = T * N
    mnb = args.num_minibatches
    mb  = B // mnb
    epochs = args.update_epochs
    gamma  = jnp.float32(args.gamma)
    lam    = jnp.float32(args.gae_lambda)
    clip_c = jnp.float32(args.clip_coef)
    ent_c  = jnp.float32(args.ent_coef)
    vf_c   = jnp.float32(args.vf_coef)

    @jax.jit
    def collect_rollout(params, state, rng):
        """T × N steps compiled into one XLA program via lax.scan."""
        def step_fn(carry, _):
            state, rng = carry
            obs = batched_obs(state)                          # (N, obs_dim)
            rng, key = jax.random.split(rng)
            keys = jax.random.split(key, N)
            action, lp, value = jax.vmap(
                lambda o, k: _policy_forward(params, o, k))(obs, keys)
            new_state, (rew, term, trunc) = batched_step(state, action)
            done = (term | trunc).astype(jnp.float32)
            return (new_state, rng), (obs, action, rew, done, lp, value)

        (final_state, rng), traj = lax.scan(
            step_fn, (state, rng), None, length=T)
        return final_state, rng, traj

    @jax.jit
    def update(params, opt_state, traj, last_value, rng):
        obs, actions, rewards, dones, old_lp, values = traj

        # GAE backward scan  (T-1 → 0)
        def gae_step(gae_acc, idx):
            r = rewards[idx]
            v = values[idx]
            # Clamp for XLA: both branches of jnp.where are always traced,
            # so the index must stay valid even when idx == T-1.
            v_next = jnp.where(idx < T - 1,
                               values[jnp.minimum(idx + 1, T - 1)],
                               last_value)
            delta = r + gamma * v_next * (1. - dones[idx]) - v
            new_gae = delta + gamma * lam * (1. - dones[idx]) * gae_acc
            return new_gae, new_gae

        _, advs_rev = lax.scan(
            gae_step,
            jnp.zeros(N, dtype=jnp.float32),
            jnp.arange(T - 1, -1, -1, dtype=jnp.int32))
        advs = advs_rev[::-1]          # (T, N)
        returns = advs + values        # (T, N)

        # Flatten to (B, ...)
        def fl(x): return x.reshape(B, *x.shape[2:])
        obs_f   = fl(obs)
        act_f   = fl(actions)
        ret_f   = returns.ravel()
        adv_f   = advs.ravel()
        lp_f    = old_lp.ravel()
        adv_f   = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        def epoch_fn(carry, _):
            params, opt_state, rng = carry
            rng, sk = jax.random.split(rng)
            idx = jax.random.permutation(sk, B)

            def mb_fn(carry, mb_i):
                params, opt_state = carry
                sel = lax.dynamic_slice(idx, (mb_i * mb,), (mb,))
                b_obs = obs_f[sel]
                b_act = act_f[sel]
                b_ret = ret_f[sel]
                b_adv = adv_f[sel]
                b_lp  = lp_f[sel]

                def loss_fn(p):
                    new_lp, ent, new_v = jax.vmap(
                        lambda o, a: _policy_evaluate(p, o, a))(b_obs, b_act)
                    ratio = jnp.exp(new_lp - b_lp)
                    pg = -jnp.minimum(
                        ratio * b_adv,
                        jnp.clip(ratio, 1. - clip_c, 1. + clip_c) * b_adv).mean()
                    vl = 0.5 * jnp.mean((new_v - b_ret) ** 2)
                    el = -ent.mean()
                    return pg + vf_c * vl + ent_c * el

                loss, grads = jax.value_and_grad(loss_fn)(params)
                updates, new_opt = optimizer.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), new_opt), loss

            (params, opt_state), losses = lax.scan(
                mb_fn, (params, opt_state),
                jnp.arange(mnb, dtype=jnp.int32))
            return (params, opt_state, rng), losses.mean()

        (params, opt_state, rng), ep_losses = lax.scan(
            epoch_fn, (params, opt_state, rng), None, length=epochs)
        return params, opt_state, rng, ep_losses.mean()

    return collect_rollout, update


# ── Training loop ──────────────────────────────────────────────────────────────


def main():
    args = tyro.cli(Args)
    B = args.num_envs * args.num_steps
    num_iters = args.total_timesteps // B
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Backend : {jax.default_backend()}"
          f"  |  envs={args.num_envs}  T={args.num_steps}  batch={B}")

    print("Loading map …")
    t0 = time.time()
    map_data = MapData(args.yaml)
    print(f"  skeleton={map_data.n_wps} pts  obs_dim={map_data.obs_dim}"
          f"  ({time.time()-t0:.1f}s)")

    batched_step, batched_obs, batched_init = make_sim_fns(
        map_data, args.max_steps, args.dr_frac)

    rng = jax.random.PRNGKey(args.seed)
    rng, init_key, policy_key = jax.random.split(rng, 3)

    # Initialise all env states
    env_keys = jax.random.split(init_key, args.num_envs)
    state = batched_init(env_keys)

    # Policy and optimizer
    params = init_policy(policy_key, map_data.obs_dim, args.hidden)
    num_iters_total = max(1, num_iters)
    lr_schedule = optax.linear_schedule(
        args.learning_rate, 0., num_iters_total * args.update_epochs * args.num_minibatches)
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(lr_schedule, eps=1e-5))
    opt_state = optimizer.init(params)

    collect_rollout, update = make_ppo_fns(
        batched_step, batched_obs, optimizer, args)

    print("Compiling (first iteration may be slow) …")
    ep_rew_buf: list[float] = []
    sps_list: list[float] = []
    global_step = 0

    for iteration in tqdm.trange(1, num_iters + 1):
        t_iter = time.time()

        rng, roll_key = jax.random.split(rng)
        state, rng, traj = collect_rollout(params, state, roll_key)
        jax.block_until_ready(traj)

        # Bootstrap value for last step
        last_obs = batched_obs(state)
        last_value = jax.vmap(
            lambda o: _mlp(params["critic"], o).squeeze())(last_obs)

        rng, upd_key = jax.random.split(rng)
        params, opt_state, rng, loss = update(
            params, opt_state, traj, last_value, upd_key)
        jax.block_until_ready(loss)

        elapsed = time.time() - t_iter
        sps = B / elapsed
        sps_list.append(sps)
        global_step += B

        _, rewards, dones, *_ = traj
        ep_rew_buf.extend(float(r) for r, d in
                          zip(rewards.ravel().tolist(), dones.ravel().tolist()) if d)

        if iteration % 10 == 0:
            mean_rew = float(np.mean(ep_rew_buf[-500:])) if ep_rew_buf else float("nan")
            print(f"  iter={iteration:6d}  "
                  f"steps={global_step/1e6:.1f}M  "
                  f"SPS={int(np.mean(sps_list[-20:])):,}  "
                  f"ep_rew={mean_rew:.1f}  "
                  f"loss={float(loss):.4f}")

        if iteration % args.save_interval == 0:
            path = os.path.join(args.save_dir, f"ckpt_{iteration}.npz")
            leaves, _ = jax.tree_util.tree_flatten(params)
            np.savez(path, *[np.array(x) for x in leaves])
            print(f"  ✓ checkpoint → {path}")

    print(f"\nDone. Mean SPS (last 20 iters): {int(np.mean(sps_list[-20:])):,}")


if __name__ == "__main__":
    main()
