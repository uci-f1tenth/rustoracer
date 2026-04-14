"""shac_jax.py — F1tenth SHAC · pure JAX · dynamic bicycle model · SLAM map.

Short Horizon Actor Critic (SHAC): gradients backprop H steps through
differentiable physics (RK4 dynamic bicycle) and differentiable reward
(bilinear-EDT wall penalty + velocity-projection track progress).
No value network or surrogate loss required.

Key differentiability choices
──────────────────────────────
• EDT (wall distance): bilinear interpolation instead of nearest-pixel, so
  ∂reward/∂(x,y) is non-zero and guides the car away from walls.
• Track progress: vx·cos(θ) + vy·sin(θ) at the nearest waypoint — smooth
  and differentiable w.r.t. the full velocity state.
• Lidar observations: sphere-traced with EDT (non-diff in xy), but used with
  lax.stop_gradient inside the SHAC unroll — they inform policy decisions
  while gradients flow exclusively through physics + reward.
• Episode resets: handled via jnp.where; the reset state is independent of
  policy params, so gradients naturally zero-out at terminal steps.

Sim-to-real: map comes from any standard ROS SLAM algorithm (yaml + pgm).
The dynamic bicycle model uses F1Tenth gym default parameters.

Usage:
    uv run --with "jax[cpu]" --with optax --with scipy --with scikit-image \\
           --with pillow --with pyyaml --with tqdm --with tyro \\
           scripts/shac_jax.py [--yaml maps/my_map.yaml]
"""
from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.ndimage as ndi
import scipy.signal
import skimage.morphology
import tqdm
import tyro
import yaml as pyyaml
from jax import lax
from PIL import Image
from scipy.spatial import KDTree

warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    yaml: str = "maps/my_map.yaml"
    """Path to SLAM map YAML (same format as ROS map_server)."""
    exp_name: str = "shac_jax"
    seed: int = 42
    num_envs: int = 0
    """Parallel environments (0 = auto: 256 on CPU, 1024 on GPU/TPU)."""
    horizon: int = 32
    """SHAC unroll length H — gradient backprops through this many physics steps."""
    total_iters: int = 10_000
    """Total gradient updates."""
    lr: float = 3e-4
    hidden: int = 256
    gamma: float = 0.99
    """Discount applied to the H-step reward sum."""
    max_grad_norm: float = 1.0
    max_steps: int = 10_000
    """Max steps per episode before forced reset."""
    dr_frac: float = 0.2
    """Domain-randomisation fraction for vehicle parameters."""
    scan_steps: int = 30
    """EDT sphere-tracing steps per lidar ray."""
    save_interval: int = 200
    """Save checkpoint every N iters."""
    save_dir: str = "checkpoints"


# ── Constants ──────────────────────────────────────────────────────────────────

STEER_MIN, STEER_MAX = np.float32(-0.5236), np.float32(0.5236)
STEER_VEL_MAX        = np.float32(3.2)
V_MIN, V_MAX         = np.float32(-5.0), np.float32(20.0)
DPSI_MAX             = np.float32(10.0)
WIDTH, LENGTH        = np.float32(0.27), np.float32(0.50)
DRAG                 = np.float32(0.5 * 0.3 * 0.04 * 1.225)
G                    = np.float32(9.81)
PI, DT               = np.float32(np.pi), np.float32(1. / 60.)
N_BEAMS, N_LOOK      = 108, 10
FOV                  = np.float32(270. * np.pi / 180.)
MIN_RANGE, MAX_RANGE = np.float32(0.06), np.float32(10.0)

# F1Tenth gym default vehicle parameters
P_DEF = {k: np.float32(v) for k, v in {
    "lf":    0.15875,
    "lr":    0.17145,
    "m":     3.74,
    "I_z":   0.04712,
    "mu":    0.523,
    "C_Sf":  4.718,
    "C_Sr":  5.4562,
    "a_max": 9.51,
}.items()}
RAND_KEYS = ("lf", "lr", "m", "a_max")


# ── Map preprocessing ──────────────────────────────────────────────────────────

def _nbrs8(p, pts):
    x, y = p
    return [(x+dx, y+dy) for dx in (-1,0,1) for dy in (-1,0,1)
            if (dx,dy) != (0,0) and (x+dx, y+dy) in pts]


def _largest_cc(pts):
    seen, best = set(), []
    for p in pts:
        if p in seen: continue
        comp, q = [], [p]; seen.add(p)
        for c in q:
            comp.append(c)
            for n in _nbrs8(c, pts):
                if n not in seen: seen.add(n); q.append(n)
        if len(comp) > len(best): best = comp
    return set(best)


def _skeleton_loop(thin, ox, oy, res, otheta, h):
    pts = set(zip(*[a.tolist() for a in np.where(thin)[::-1]]))
    if not pts: raise RuntimeError("Empty skeleton")
    pts = _largest_cc(pts)
    for _ in range(500):
        ends = [p for p in pts if len(_nbrs8(p, pts)) < 2]
        if not ends: break
        pts -= set(ends)
    if not pts: raise RuntimeError("Skeleton collapsed")
    start = min(pts, key=lambda p: (p[0]+ox/res)**2 + (p[1]-(h-1)-oy/res)**2)
    adj = _nbrs8(start, pts)
    if len(adj) < 2: raise RuntimeError("Start node < 2 neighbours")
    src, targets = adj[0], set(adj[1:])
    parent, queue, found = {src: src}, [src], None
    for cur in queue:
        for n in _nbrs8(cur, pts):
            if n == start or n in parent: continue
            parent[n] = cur
            if n in targets: found = n; break
            queue.append(n)
        if found: break
    path = [start]
    if found:
        p = found
        while p != src: path.append(p); p = parent[p]
        path.append(src)
    path.reverse()
    world = [[px*res+ox, (h-1-py)*res+oy] for px,py in path]
    if len(world) >= 2:
        dx0, dy0 = world[1][0]-world[0][0], world[1][1]-world[0][1]
        if abs(((np.arctan2(dy0,dx0)-otheta+3*np.pi)%(2*np.pi))-np.pi) > np.pi/2:
            world[1:] = world[1:][::-1]
    return world


class MapData:
    """Preprocessed map: EDT, centreline skeleton, lidar beam table, RGB image."""

    def __init__(self, yaml_path: str,
                 edt_thresh_frac: float = 0.1, sg_window: int = 11, sg_degree: int = 3):
        meta   = pyyaml.safe_load(open(yaml_path))
        raw    = np.array(Image.open(Path(yaml_path).parent / meta["image"]).convert("L"))
        res    = float(meta["resolution"])
        ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
        otheta = float(meta["origin"][2]) if len(meta["origin"]) > 2 else 0.
        h, w   = raw.shape

        free    = raw >= 210
        edt_np  = ndi.distance_transform_edt(free).astype(np.float32) * res
        edt_px  = ndi.distance_transform_edt(free)
        loop    = _skeleton_loop(
            skimage.morphology.skeletonize(free & (edt_px >= edt_px.max()*edt_thresh_frac)),
            ox, oy, res, otheta, h)
        assert len(loop) >= 2
        skel = np.array(scipy.signal.savgol_filter(
            np.array(loop, np.float64), sg_window, sg_degree, axis=0, mode="wrap"),
            dtype=np.float32)

        ys, xs = np.mgrid[0:h, 0:w]
        wx = (xs.ravel() * res + ox).astype(np.float32)
        wy = ((h-1-ys.ravel()) * res + oy).astype(np.float32)
        _, lut = KDTree(skel).query(np.c_[wx, wy])

        angles = np.linspace(-FOV/2, FOV/2, N_BEAMS, dtype=np.float32)
        self.edt      = jnp.array(edt_np)
        self.skeleton = jnp.array(skel)
        self.skel_lut = jnp.array(lut.astype(np.int32))
        self.beam_sc  = jnp.array(np.stack([np.sin(angles), np.cos(angles)], 1))
        self.w, self.h    = w, h
        self.res          = np.float32(res)
        self.inv_res      = np.float32(1. / res)
        self.ox, self.oy  = np.float32(ox), np.float32(oy)
        self.n_wps        = len(skel)
        seg_len = np.linalg.norm(skel[1:]-skel[:-1], axis=1).mean()
        self.look_step    = max(1, round(1./seg_len)) if seg_len > 0 else 1
        self.obs_dim      = N_BEAMS + 3 + 2 + N_LOOK * 2


# ── Dynamic bicycle model ──────────────────────────────────────────────────────
# State s = [x, y, δ, v_x, v_y, ψ, ψ̇]

def _dyn(s, accel, sv, p):
    """Single-track dynamic bicycle model with linearised Pacejka tyre forces."""
    x, y, delta, vx, vy, psi, dpsi = s
    lwb     = p["lf"] + p["lr"]
    safe_vx = jnp.maximum(jnp.abs(vx), jnp.float32(0.05))

    # Slip angles — arctan valid since safe_vx > 0
    alpha_f = jnp.arctan((dpsi * p["lf"] + vy) / safe_vx) - delta
    alpha_r = jnp.arctan((dpsi * p["lr"] - vy) / safe_vx)

    mg   = p["m"] * G
    F_yf = -p["mu"] * mg * (p["lr"] / lwb) * p["C_Sf"] * alpha_f
    F_yr = -p["mu"] * mg * (p["lf"] / lwb) * p["C_Sr"] * alpha_r
    F_x  = p["m"] * accel - DRAG * vx * jnp.abs(vx)

    return jnp.array([
        vx * jnp.cos(psi) - vy * jnp.sin(psi),
        vx * jnp.sin(psi) + vy * jnp.cos(psi),
        sv,
        (F_x - F_yf * jnp.sin(delta) + p["m"] * vy * dpsi) / p["m"],
        (F_yr + F_yf * jnp.cos(delta) - p["m"] * vx * dpsi) / p["m"],
        dpsi,
        (F_yf * p["lf"] * jnp.cos(delta) - F_yr * p["lr"]) / p["I_z"],
    ])


# ── Simulation ─────────────────────────────────────────────────────────────────

def make_shac_fns(map_data: MapData, args: Args):
    """Return (shac_loss_fn, batched_init, batched_step).

    shac_loss_fn(params, states) → scalar loss (differentiable).
    batched_init(rngs)           → initial state dict for N envs.
    batched_step(state, action)  → (new_state, (rew, term, trunc)).
    """
    edt, skel, lut, bsc = map_data.edt, map_data.skeleton, map_data.skel_lut, map_data.beam_sc
    W, H       = map_data.w, map_data.h
    res, inv   = map_data.res, map_data.inv_res
    ox, oy     = map_data.ox, map_data.oy
    n_wps      = jnp.int32(map_data.n_wps)
    look_step  = jnp.int32(map_data.look_step)
    dt_f       = jnp.float32(float(DT))
    Wf, Hf     = jnp.float32(W), jnp.float32(H)
    COLR       = jnp.float32(np.sqrt((LENGTH/2)**2 + (WIDTH/2)**2))
    H_steps    = args.horizon
    gamma_f    = jnp.float32(args.gamma)
    dr_frac    = args.dr_frac
    max_steps  = args.max_steps
    scan_steps = args.scan_steps

    def _to_px(x, y):
        return (jnp.int32(jnp.clip((x-ox)*inv,        0., Wf-1.)),
                jnp.int32(jnp.clip((Hf-1.)-(y-oy)*inv, 0., Hf-1.)))

    def _edt_bl(x, y):
        """Bilinear-interpolated EDT value at world position (x, y).

        Differentiable w.r.t. x and y, giving smooth wall-avoidance gradients.
        """
        pxf = (x - ox) * inv
        pyf = (Hf - 1.) - (y - oy) * inv
        x0  = jnp.floor(pxf).astype(jnp.int32)
        y0  = jnp.floor(pyf).astype(jnp.int32)
        x0c = jnp.clip(x0,   0, W-1); x1c = jnp.clip(x0+1, 0, W-1)
        y0c = jnp.clip(y0,   0, H-1); y1c = jnp.clip(y0+1, 0, H-1)
        fx  = pxf - jnp.floor(pxf); fy = pyf - jnp.floor(pyf)
        return (edt[y0c, x0c] * (1-fx) * (1-fy)
              + edt[y0c, x1c] * fx     * (1-fy)
              + edt[y1c, x0c] * (1-fx) * fy
              + edt[y1c, x1c] * fx     * fy)

    def _collides(s):
        x, y = s[0], s[1]
        pxf, pyf = (x-ox)*inv, (Hf-1.)-(y-oy)*inv
        in_map   = (pxf>=0.)&(pxf<Wf)&(pyf>=0.)&(pyf<Hf)
        d = jnp.where(in_map, edt[jnp.int32(jnp.clip(pyf,0.,Hf-1.)),
                                    jnp.int32(jnp.clip(pxf,0.,Wf-1.))], 0.)
        return (~jnp.isfinite(x)) | (~jnp.isfinite(y)) | (d < COLR)

    def _obs(s, wi, p):
        x, y, delta, vx, vy, psi, dpsi = s
        sh, ch = jnp.sin(psi), jnp.cos(psi)
        sa, ca  = bsc[:,0], bsc[:,1]
        bdx, bdy = ch*ca - sh*sa, sh*ca + ch*sa
        px0 = (x + p["lf"]*ch - ox)*inv;  py0 = (Hf-1.) - (y + p["lf"]*sh - oy)*inv
        dpx, dpy = bdx*inv, -bdy*inv
        minstep = jnp.float32(res * 0.5)
        def _ray(carry, _):
            t, r, done = carry
            pxf, pyf = px0 + t*dpx, py0 + t*dpy
            ib = (pxf>=0.)&(pxf<Wf)&(pyf>=0.)&(pyf<Hf)
            d  = jnp.where(ib, edt[jnp.int32(jnp.clip(pyf,0.,Hf-1.)),
                                    jnp.int32(jnp.clip(pxf,0.,Wf-1.))], 0.)
            hit  = (~done) & (~ib | (d < minstep))
            r    = jnp.where(hit, t, r)
            done = done | hit
            t    = jnp.where(done, t, t + jnp.maximum(d, minstep))
            return (t, r, done), None
        (_, scans, _), _ = lax.scan(_ray, (jnp.full(N_BEAMS, MIN_RANGE),
                                            jnp.full(N_BEAMS, MAX_RANGE),
                                            jnp.zeros(N_BEAMS, jnp.bool_)),
                                    None, length=scan_steps)
        wp, wpn  = skel[wi], skel[(wi+1)%n_wps]
        cth      = jnp.arctan2(wpn[1]-wp[1], wpn[0]-wp[0])
        he       = ((cth-psi)+PI) % (2.*PI) - PI
        le       = -(x-wp[0])*jnp.sin(cth) + (y-wp[1])*jnp.cos(cth)
        idx  = (wi + jnp.arange(1, N_LOOK+1, dtype=jnp.int32)*look_step) % n_wps
        dxy  = skel[idx] - jnp.array([x, y])
        look = jnp.stack([dxy[:,0]*ch+dxy[:,1]*sh, -dxy[:,0]*sh+dxy[:,1]*ch], 1).ravel()
        return jnp.concatenate([scans, jnp.array([vx, delta, dpsi]),
                                  jnp.array([he, le]), look])

    def _reset(rng):
        rng, sub = jax.random.split(rng)
        ri  = jax.random.randint(jax.random.fold_in(sub, jnp.uint32(0)), (), 0, n_wps)
        pt, ptn = skel[ri], skel[(ri+1)%n_wps]
        p = {k: P_DEF[k] for k in P_DEF}
        for i, k in enumerate(RAND_KEYS):
            u    = jax.random.uniform(jax.random.fold_in(sub, jnp.uint32(i+1)))
            p[k] = P_DEF[k] * (1. + jnp.float32(dr_frac) * (u * 2. - 1.))
        th = jnp.arctan2(ptn[1]-pt[1], ptn[0]-pt[0])
        s  = jnp.array([pt[0], pt[1], jnp.float32(0.), jnp.float32(0.),
                         jnp.float32(0.), th, jnp.float32(0.)])
        return s, ri, p, rng

    def _step(state, action):
        s, wi, steps, rng, p = (state["dyn"], state["wp_idx"],
                                 state["steps"], state["rng"], state["p"])
        sv    = jnp.clip(action[0], -1., 1.) * STEER_VEL_MAX
        sv    = jnp.where((sv<0.)&(s[2]<=STEER_MIN), 0., sv)
        sv    = jnp.where((sv>0.)&(s[2]>=STEER_MAX), 0., sv)
        accel = jnp.clip(action[1], -1., 1.) * p["a_max"]
        f  = lambda s: _dyn(s, accel, sv, p)
        k1 = f(s); k2 = f(s+k1*(dt_f/2)); k3 = f(s+k2*(dt_f/2)); k4 = f(s+k3*dt_f)
        s  = s + dt_f/6. * (k1 + 2.*k2 + 2.*k3 + k4)
        x, y, delta, vx, vy, psi, dpsi = s
        s  = jnp.array([x, y,
                         jnp.clip(delta, STEER_MIN, STEER_MAX),
                         jnp.clip(vx, V_MIN, V_MAX),
                         jnp.clip(vy, -V_MAX, V_MAX),
                         (psi+PI)%(2.*PI)-PI,
                         jnp.clip(dpsi, -DPSI_MAX, DPSI_MAX)])
        px, py  = _to_px(x, y)
        new_wi  = lut[py*W + px]

        # ── Differentiable reward ──────────────────────────────────────────────
        # Wall penalty via bilinear EDT (differentiable w.r.t. x, y)
        edt_val  = _edt_bl(s[0], s[1])
        wall_pen = jnp.exp(-3. * jnp.maximum(edt_val, jnp.float32(0.)))
        # Track progress: velocity projection onto centreline tangent
        wp       = skel[new_wi]; wpn = skel[(new_wi+1)%n_wps]
        cth      = jnp.arctan2(wpn[1]-wp[1], wpn[0]-wp[0])
        progress = s[3]*jnp.cos(cth) + s[4]*jnp.sin(cth)   # vx·cos + vy·sin
        le       = -(s[0]-wp[0])*jnp.sin(cth) + (s[1]-wp[1])*jnp.cos(cth)
        reward   = (progress * (1. + jnp.maximum(s[3], jnp.float32(0.)) / 10.)
                    - 5. * wall_pen
                    - 0.05 * le ** 2)

        term    = _collides(s)
        trunc   = (steps + jnp.int32(1)) >= jnp.int32(max_steps)
        reward -= jnp.where(term, jnp.float32(50.), jnp.float32(0.))

        rs, rwi, rp, rng = _reset(rng)
        done = term | trunc
        s   = jnp.where(done, rs, s);   wi = jnp.where(done, rwi, new_wi)
        p   = jax.tree.map(lambda a, b: jnp.where(done, a, b), rp, p)
        st  = jnp.where(done, jnp.int32(0), steps + jnp.int32(1))
        ns  = {"dyn": s, "wp_idx": wi, "steps": st, "rng": rng, "p": p,
               "obs": _obs(s, wi, p)}
        return ns, (reward, term, trunc)

    def _init(rng):
        s, ri, p, rng = _reset(rng)
        return {"dyn": s, "wp_idx": ri, "steps": jnp.int32(0), "rng": rng, "p": p,
                "obs": _obs(s, ri, p)}

    batched_step = jax.jit(jax.vmap(_step))
    batched_init = jax.jit(jax.vmap(_init))

    def shac_loss(params, states):
        """SHAC loss: negative discounted H-step reward sum.

        Gradients backprop through physics (_step) and the differentiable reward
        (bilinear EDT + velocity projection). Lidar inside _obs is stop_gradient'd
        at each step so the physics chain is the sole gradient path.
        """
        disc = gamma_f ** jnp.arange(H_steps, dtype=jnp.float32)

        def scan_fn(state, d):
            # Observations are treated as constants in the gradient graph —
            # they guide action selection but gradients flow via physics only.
            obs = lax.stop_gradient(state["obs"])
            act = jax.vmap(lambda o: _mlp(params, o))(obs)
            new_state, (rew, _term, _trunc) = batched_step(state, act)
            return new_state, rew * d

        final_states, rews = lax.scan(scan_fn, states, disc)
        return -jnp.mean(rews), final_states

    return shac_loss, batched_init, batched_step


# ── Policy ─────────────────────────────────────────────────────────────────────

def init_policy(key, obs_dim: int, hidden: int, act_dim: int = 2):
    def layer(k, i, o, std=float(np.sqrt(2))):
        return {"w": jax.random.normal(k, (i, o)) * float(std / np.sqrt(i)),
                "b": jnp.zeros(o)}
    ks = jax.random.split(key, 4)
    return [layer(ks[0], obs_dim, hidden),
            layer(ks[1], hidden, hidden),
            layer(ks[2], hidden, hidden),
            layer(ks[3], hidden, act_dim, std=0.01)]


def _mlp(params, x):
    for l in params[:-1]: x = jax.nn.tanh(x @ l["w"] + l["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


# ── Training ───────────────────────────────────────────────────────────────────

def main():
    args = tyro.cli(Args)
    if args.num_envs <= 0:
        args.num_envs = 256 if jax.default_backend() == "cpu" else 1024
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Backend: {jax.default_backend()}  "
          f"envs={args.num_envs}  H={args.horizon}")
    t0 = time.time()
    map_data = MapData(args.yaml)
    print(f"  map: {map_data.n_wps} wps  obs_dim={map_data.obs_dim}  "
          f"({time.time()-t0:.1f}s)")

    shac_loss, batched_init, batched_step = make_shac_fns(map_data, args)

    rng  = jax.random.PRNGKey(args.seed)
    rng, ki, kp = jax.random.split(rng, 3)
    states = batched_init(jax.random.split(ki, args.num_envs))
    params = init_policy(kp, map_data.obs_dim, args.hidden)

    opt       = optax.chain(optax.clip_by_global_norm(args.max_grad_norm),
                             optax.adam(args.lr))
    opt_state = opt.init(params)

    @jax.jit
    def update(params, opt_state, states):
        (loss, new_states), grads = jax.value_and_grad(shac_loss, has_aux=True)(
            params, states)
        ups, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, ups)
        return params, opt_state, lax.stop_gradient(new_states), loss

    print("Compiling first update… (may take 1–5 min on CPU)")
    t_compile = time.time()
    params, opt_state, states, loss = update(params, opt_state, states)
    jax.block_until_ready(loss)
    print(f"  compiled in {time.time()-t_compile:.1f}s  "
          f"loss={float(loss):.4f}")

    sps_buf: list[float] = []
    speeds:  list[float] = []
    global_step = args.num_envs * args.horizon  # already did one update

    for it in tqdm.trange(2, args.total_iters + 1):
        t_it = time.time()
        params, opt_state, states, loss = update(params, opt_state, states)
        jax.block_until_ready(loss)

        steps_this = args.num_envs * args.horizon
        sps = steps_this / (time.time() - t_it)
        sps_buf.append(sps)
        global_step += steps_this

        mean_v = float(jnp.mean(states["dyn"][:, 3]))
        speeds.append(mean_v)

        if it % 50 == 0:
            print(f"  iter={it:6d}  steps={global_step/1e6:.2f}M  "
                  f"SPS={int(np.mean(sps_buf[-20:])):,}  "
                  f"loss={float(loss):.4f}  "
                  f"mean_vx={float(np.mean(speeds[-20:])):.2f} m/s")

        if it % args.save_interval == 0:
            ckpt = os.path.join(args.save_dir, f"shac_ckpt_{it}.npz")
            leaves, _ = jax.tree_util.tree_flatten(params)
            np.savez(ckpt, *[np.array(x) for x in leaves])
            print(f"  ✓ {ckpt}")

    print(f"\nDone. Mean SPS: {int(np.mean(sps_buf[-20:])):,}")


if __name__ == "__main__":
    main()
