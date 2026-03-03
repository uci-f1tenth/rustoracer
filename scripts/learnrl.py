from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
import wandb
from torch.distributions.normal import Normal

from rustoracerpy import RustoracerEnv


# ━━━━━━━━━━━━━━━━━━━━  hyper-parameters  ━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = True
    """whether to capture videos of the agent performances"""

    # Algorithm specific arguments
    yaml: str = "maps/berlin.yaml"
    """path to the RustoracerEnv YAML map file"""
    total_timesteps: int = 50_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1024
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    max_ep_steps: int = 10_000
    """maximum steps per episode"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.02
    """the target KL divergence threshold"""

    # Network
    hidden: int = 256
    """hidden layer size"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    video_interval: int = 25
    """record an evaluation video every N iterations (0 to disable)"""
    video_max_steps: int = 600
    """max steps per evaluation video episode"""

    save_interval: int = 50
    """save checkpoint every N iterations"""
    save_dir: str = "checkpoints"
    """directory to save checkpoints"""

    compile: bool = False
    """whether to use torch.compile"""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile"""


# ━━━━━━━━━━━━  GPU running mean / std (Welford, f32 accumulators)  ━━━━━━━━━━━
class GPURunningMeanStd:
    """Welford running stats kept on device."""

    def __init__(self, shape: tuple[int, ...], device: torch.device) -> None:
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count: float = 1e-4
        self.device = device
        # Cached for fast normalize
        self._inv_std = torch.ones(shape, dtype=torch.float32, device=device)

    def _refresh_cache(self) -> None:
        self._inv_std = torch.rsqrt(self.var + 1e-8)

    def update(self, batch: torch.Tensor) -> None:
        batch = batch.reshape(-1, *self.mean.shape).float()
        bm = batch.mean(0)
        bv = batch.var(0, correction=0)
        bc = batch.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        new_mean = self.mean + delta * (bc / tot)
        m2 = self.var * self.count + bv * bc + delta**2 * (self.count * bc / tot)
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot
        self._refresh_cache()

    def normalize(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        return ((x - self.mean) * self._inv_std).clamp(-clip, clip)


class GPURunningReturnStd:
    """Track discounted return variance for reward normalisation."""

    def __init__(self, num_envs: int, gamma: float, device: torch.device) -> None:
        self.gamma = gamma
        self.returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.rms = GPURunningMeanStd(shape=(), device=device)

    def update(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        self.returns = (
            self.returns * self.gamma * (1.0 - dones.float()) + rewards.float()
        )
        self.rms.update(self.returns)

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        return rewards * self.rms._inv_std


# ━━━━━━━━━━  CPU running mean / std (for eval video only)  ━━━━━━━━━━━
class CPURunningMeanStd:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count: float = 1e-4

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)


# ━━━━━━━━━━━━━━━━━━  actor-critic network  ━━━━━━━━━━━━━━━━━━━━━━
LOG_STD_MIN = -5.0
LOG_STD_MAX = 0.5


def layer_init(
    layer: nn.Linear, std: float = float(np.sqrt(2)), bias: float = 0.0
) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)  # ← needs QR, CPU-only on MPS
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        # Build + orthogonal-init on CPU (avoids MPS linalg_qr gap),
        # caller does .to(device) afterward.
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), -0.5))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(x)
        log_std = self.actor_logstd.expand_as(mean).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mean, std)
        if action is None:
            action = mean + std * torch.randn_like(mean)
        return (
            action,
            dist.log_prob(action).sum(-1),
            dist.entropy().sum(-1),
            self.critic(x).squeeze(-1),
        )


# ─────────────────────────────────────────────────────────────
# Video recording helper
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def record_eval_video(
    yaml: str,
    agent: Agent,
    obs_rms_cpu: CPURunningMeanStd,
    device: torch.device,
    max_steps: int = 600,
    max_ep_steps: int = 10_000,
) -> Tuple["wandb.Video | None", float]:
    eval_env = RustoracerEnv(
        yaml=yaml, num_envs=1, max_steps=max_ep_steps, render_mode="rgb_array"
    )
    raw_obs, _ = eval_env.reset(seed=42)
    frames: list[np.ndarray] = []
    total_reward = 0.0
    for _ in range(max_steps):
        frame = eval_env.render()
        if frame is not None:
            frames.append(frame)
        obs_norm = obs_rms_cpu.normalize(raw_obs)
        obs_t = torch.tensor(obs_norm, device=device, dtype=torch.float32)
        action_mean = agent.actor_mean(obs_t)
        action_np = action_mean.cpu().numpy().astype(np.float64).clip(-1.0, 1.0)
        raw_obs, reward, terminated, truncated, _ = eval_env.step(action_np)
        total_reward += float(reward[0])
        if terminated[0] or truncated[0]:
            break
    eval_env.close()
    if not frames:
        return None, total_reward
    video_np = np.stack(frames).transpose(0, 3, 1, 2)
    return wandb.Video(video_np, fps=60, format="mp4"), total_reward


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def gpu_rms_to_cpu(gpu_rms: GPURunningMeanStd) -> CPURunningMeanStd:
    cpu_rms = CPURunningMeanStd(shape=tuple(gpu_rms.mean.shape))
    cpu_rms.mean = gpu_rms.mean.cpu().numpy()
    cpu_rms.var = gpu_rms.var.cpu().numpy()
    cpu_rms.count = gpu_rms.count
    return cpu_rms


def save_checkpoint(
    args: Args,
    agent: Agent,
    optimizer: optim.Adam,
    obs_rms: GPURunningMeanStd,
    ret_rms: GPURunningReturnStd,
    global_step: int,
    update: int,
    name: str | None = None,
) -> None:
    if name is None:
        name = f"agent_{global_step}.pt"
    path = os.path.join(args.save_dir, name)
    torch.save(
        {
            "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "obs_rms_mean": obs_rms.mean.cpu(),
            "obs_rms_var": obs_rms.var.cpu(),
            "obs_rms_count": obs_rms.count,
            "ret_rms_var": ret_rms.rms.var.cpu(),
            "ret_rms_count": ret_rms.rms.count,
            "global_step": global_step,
            "update": update,
        },
        path,
    )
    print(f"  💾  {path}")


# ─────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = tyro.cli(Args)

    batch_size = args.num_envs * args.num_steps
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = batch_size
    args.num_iterations = args.total_timesteps // batch_size
    run_name = f"Rustoracer__{args.exp_name}__{args.seed}"

    print(f"batch_size        = {args.batch_size:,}")
    print(f"minibatch_size    = {args.minibatch_size:,}")
    print(f"num_iterations    = {args.num_iterations}")
    print(f"hidden            = {args.hidden}")
    print(f"update_epochs     = {args.update_epochs}")
    print(f"ent_coef          = {args.ent_coef}")
    print(f"target_kl         = {args.target_kl}")
    print(f"max_ep_steps      = {args.max_ep_steps}")
    print(f"compile           = {args.compile}")
    print(f"cudagraphs        = {args.cudagraphs}")

    wandb.init(
        project="ppo_continuous_action",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )

    # ── seeding ──────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # ── device ───────────────────────────────────────────────────
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device : {device}")

    is_cuda = device.type == "cuda"

    # ── environment ──────────────────────────────────────────────
    env = RustoracerEnv(
        yaml=args.yaml, num_envs=args.num_envs, max_steps=args.max_ep_steps
    )
    obs_dim: int = env.single_observation_space.shape[0]
    act_dim: int = env.single_action_space.shape[0]
    print(f"obs={obs_dim}  act={act_dim}  envs={args.num_envs}")

    # ── network + optimiser ──────────────────────────────────────
    agent = Agent(obs_dim, act_dim, args.hidden).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    print(f"Parameters : {sum(p.numel() for p in agent.parameters()):,}")

    # ── GPU normalisers ──────────────────────────────────────────
    obs_rms = GPURunningMeanStd(shape=(obs_dim,), device=device)
    ret_rms = GPURunningReturnStd(args.num_envs, args.gamma, device=device)

    # ── GPU rollout buffers (float32) ────────────────────────────
    obs_buf = torch.zeros(
        (args.num_steps, args.num_envs, obs_dim), dtype=torch.float32, device=device
    )
    act_buf = torch.zeros(
        (args.num_steps, args.num_envs, act_dim), dtype=torch.float32, device=device
    )
    logp_buf = torch.zeros(
        (args.num_steps, args.num_envs), dtype=torch.float32, device=device
    )
    rew_buf = torch.zeros(
        (args.num_steps, args.num_envs), dtype=torch.float32, device=device
    )
    done_buf = torch.zeros(
        (args.num_steps, args.num_envs), dtype=torch.float32, device=device
    )
    val_buf = torch.zeros(
        (args.num_steps, args.num_envs), dtype=torch.float32, device=device
    )

    # ── Pinned CPU staging buffers for fast async transfer ───────
    if is_cuda:
        pin_obs = torch.zeros(
            (args.num_envs, obs_dim), dtype=torch.float32
        ).pin_memory()
        pin_rew = torch.zeros((args.num_envs,), dtype=torch.float32).pin_memory()
        pin_done = torch.zeros((args.num_envs,), dtype=torch.float32).pin_memory()
    else:
        pin_obs = pin_rew = pin_done = None

    # ── episode trackers (CPU — avoids GPU syncs) ────────────────
    ep_ret_cpu = np.zeros(args.num_envs, dtype=np.float64)
    ep_len_cpu = np.zeros(args.num_envs, dtype=np.int64)
    recent_returns: list[float] = []
    recent_lengths: list[int] = []

    # ── Reusable CPU buffer for clipped actions ──────────────────
    act_np_buf = np.empty((args.num_envs, act_dim), dtype=np.float64)

    # ── first reset ──────────────────────────────────────────────
    next_obs_raw_np, _ = env.reset(seed=args.seed)

    if is_cuda:
        pin_obs.numpy()[:] = next_obs_raw_np.astype(np.float32)
        next_obs_raw = pin_obs.to(device, non_blocking=True).clone()
    else:
        next_obs_raw = torch.as_tensor(
            next_obs_raw_np, dtype=torch.float32, device=device
        )

    obs_rms.update(next_obs_raw)
    next_obs_n = obs_rms.normalize(next_obs_raw)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

    os.makedirs(args.save_dir, exist_ok=True)
    global_step = 0
    t0 = time.time()

    # ── compilable functions ─────────────────────────────────────
    _gamma = args.gamma
    _gae_lambda = args.gae_lambda
    _num_steps = args.num_steps
    _clip_coef = args.clip_coef
    _vf_coef = args.vf_coef
    _ent_coef = args.ent_coef
    _max_grad_norm = args.max_grad_norm
    _clip_vloss = args.clip_vloss
    _norm_adv = args.norm_adv

    def compute_gae(
        next_val: torch.Tensor,
        next_done_t: torch.Tensor,
        rew_buf: torch.Tensor,
        done_buf: torch.Tensor,
        val_buf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rew_buf)
        lastgae = torch.zeros_like(next_val)
        for t in range(_num_steps - 1, -1, -1):
            if t == _num_steps - 1:
                nnt = 1.0 - next_done_t
                nv = next_val
            else:
                nnt = 1.0 - done_buf[t + 1]
                nv = val_buf[t + 1]
            delta = rew_buf[t] + _gamma * nv * nnt - val_buf[t]
            lastgae = delta + _gamma * _gae_lambda * nnt * lastgae
            advantages[t] = lastgae
        returns = advantages + val_buf
        return advantages, returns

    def ppo_update_step(
        b_obs: torch.Tensor,
        b_act: torch.Tensor,
        b_logp: torch.Tensor,
        b_adv: torch.Tensor,
        b_ret: torch.Tensor,
        b_val: torch.Tensor,
        mb_inds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, newlogp, entropy, newval = agent.get_action_and_value(
            b_obs[mb_inds], b_act[mb_inds]
        )
        logratio = newlogp - b_logp[mb_inds]
        ratio = logratio.exp()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > _clip_coef).float().mean()

        mb_adv = b_adv[mb_inds]
        if _norm_adv:
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

        # Policy loss
        pg1 = -mb_adv * ratio
        pg2 = -mb_adv * ratio.clamp(1 - _clip_coef, 1 + _clip_coef)
        pg_loss = torch.max(pg1, pg2).mean()

        # Value loss
        if _clip_vloss:
            v_unclipped = (newval - b_ret[mb_inds]) ** 2
            v_clipped = b_val[mb_inds] + (newval - b_val[mb_inds]).clamp(
                -_clip_coef, _clip_coef
            )
            v_loss = (
                0.5 * torch.max(v_unclipped, (v_clipped - b_ret[mb_inds]) ** 2).mean()
            )
        else:
            v_loss = 0.5 * ((newval - b_ret[mb_inds]) ** 2).mean()

        ent_loss = entropy.mean()
        loss = pg_loss - _ent_coef * ent_loss + _vf_coef * v_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), _max_grad_norm)
        optimizer.step()

        return (
            pg_loss.detach(),
            v_loss.detach(),
            ent_loss.detach(),
            approx_kl.detach(),
            clipfrac.detach(),
        )

    if args.compile:
        compute_gae = torch.compile(compute_gae, fullgraph=True)
        ppo_update_step = torch.compile(ppo_update_step)

    # ── video iteration schedule ─────────────────────────────────
    if args.capture_video and args.video_interval > 0:
        video_iters = set(range(1, args.num_iterations + 1, args.video_interval))
        video_iters.add(args.num_iterations)
    else:
        video_iters = set()

    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))

    # ═══════════════════  outer loop  ════════════════════════════
    for update in pbar:
        # ── learning-rate annealing ──────────────────────────────
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / args.num_iterations
            optimizer.param_groups[0]["lr"] = args.learning_rate * frac

        # ── rollout collection ───────────────────────────────────
        for step in range(args.num_steps):
            global_step += args.num_envs

            obs_buf[step] = next_obs_n
            done_buf[step] = next_done

            with torch.inference_mode():
                action, logprob, _, value = agent.get_action_and_value(next_obs_n)

            val_buf[step] = value
            logp_buf[step] = logprob
            act_buf[step] = action

            # Transfer action to CPU for env.step (unavoidable sync)
            np.copyto(act_np_buf, action.cpu().numpy().clip(-1.0, 1.0))

            next_obs_raw_np, reward_np, terminated_np, truncated_np, _ = env.step(
                act_np_buf
            )
            done_np = np.logical_or(terminated_np, truncated_np).astype(np.float32)

            # Fast CPU→GPU via pinned memory
            if is_cuda:
                pin_obs.numpy()[:] = next_obs_raw_np.astype(np.float32)
                next_obs_raw = pin_obs.to(device, non_blocking=True)
                pin_rew.numpy()[:] = reward_np.astype(np.float32)
                reward_t = pin_rew.to(device, non_blocking=True)
                pin_done.numpy()[:] = done_np
                done_t = pin_done.to(device, non_blocking=True)
            else:
                next_obs_raw = torch.as_tensor(
                    next_obs_raw_np.astype(np.float32), device=device
                )
                reward_t = torch.as_tensor(reward_np.astype(np.float32), device=device)
                done_t = torch.as_tensor(done_np, device=device)

            # Reward normalisation
            ret_rms.update(reward_t, done_t)
            rew_buf[step] = ret_rms.normalize(reward_t)

            next_done = done_t

            # Episode bookkeeping (CPU — no GPU syncs)
            ep_ret_cpu += reward_np
            ep_len_cpu += 1
            done_idx = np.where(done_np > 0.5)[0]
            for i in done_idx:
                recent_returns.append(float(ep_ret_cpu[i]))
                recent_lengths.append(int(ep_len_cpu[i]))
                ep_ret_cpu[i] = 0.0
                ep_len_cpu[i] = 0

            # Update obs normaliser
            obs_rms.update(next_obs_raw)
            next_obs_n = obs_rms.normalize(next_obs_raw)

        # ── GAE ──────────────────────────────────────────────────
        with torch.inference_mode():
            next_val = agent.get_value(next_obs_n).squeeze(-1)

        advantages, returns = compute_gae(
            next_val, next_done, rew_buf, done_buf, val_buf
        )

        # ── flatten ──────────────────────────────────────────────
        b_obs = obs_buf.reshape(batch_size, obs_dim)
        b_act = act_buf.reshape(batch_size, act_dim)
        b_logp = logp_buf.reshape(batch_size)
        b_adv = advantages.reshape(batch_size)
        b_ret = returns.reshape(batch_size)
        b_val = val_buf.reshape(batch_size)

        # ── PPO update epochs ────────────────────────────────────
        # Accumulators on GPU — sync once at end
        sum_pg = torch.zeros((), device=device)
        sum_vl = torch.zeros((), device=device)
        sum_ent = torch.zeros((), device=device)
        sum_kl = torch.zeros((), device=device)
        sum_cf = torch.zeros((), device=device)
        num_updates = 0
        kl_early_stopped = False

        for _epoch in range(args.update_epochs):
            if kl_early_stopped:
                break
            b_inds = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]

                pg_l, v_l, e_l, kl_l, cf_l = ppo_update_step(
                    b_obs, b_act, b_logp, b_adv, b_ret, b_val, mb_inds
                )

                sum_pg += pg_l
                sum_vl += v_l
                sum_ent += e_l
                sum_kl += kl_l
                sum_cf += cf_l
                num_updates += 1

            # Per-epoch KL check (only 1 sync per epoch, not per minibatch)
            if kl_l.item() > args.target_kl * 1.5:
                kl_early_stopped = True

        # ── single GPU→CPU sync for all metrics ─────────────────
        if num_updates > 0:
            inv = 1.0 / num_updates
            metrics = (
                torch.stack([sum_pg, sum_vl, sum_ent, sum_kl, sum_cf]).cpu().numpy()
                * inv
            )
            pg_loss_val, v_loss_val, ent_loss_val, approx_kl_val, clipfrac_val = metrics
        else:
            pg_loss_val = v_loss_val = ent_loss_val = approx_kl_val = clipfrac_val = 0.0

        # ── logging ──────────────────────────────────────────────
        elapsed = time.time() - t0
        sps = global_step / elapsed
        mr = float(np.mean(recent_returns[-200:])) if recent_returns else 0.0
        ml = float(np.mean(recent_lengths[-200:])) if recent_lengths else 0.0

        log_dict = {
            "charts/episode_return": mr,
            "charts/episode_length": ml,
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "charts/SPS": sps,
            "losses/pg_loss": pg_loss_val,
            "losses/v_loss": v_loss_val,
            "losses/entropy": ent_loss_val,
            "losses/approx_kl": approx_kl_val,
            "losses/clipfrac": clipfrac_val,
            "debug/kl_early_stopped": int(kl_early_stopped),
            "perf/global_step": global_step,
        }

        # ── video ────────────────────────────────────────────────
        if update in video_iters:
            print(f"\n[iter {update}] Recording evaluation video...")
            agent.eval()
            obs_rms_cpu = gpu_rms_to_cpu(obs_rms)
            vid, eval_reward = record_eval_video(
                yaml=args.yaml,
                agent=agent,
                obs_rms_cpu=obs_rms_cpu,
                device=device,
                max_steps=args.video_max_steps,
                max_ep_steps=args.max_ep_steps,
            )
            agent.train()
            if vid is not None:
                log_dict["media/eval_video"] = vid
                print(f"[iter {update}] Video captured, eval_return={eval_reward:.2f}")
            else:
                print(f"[iter {update}] WARNING: no frames captured!")
            log_dict["charts/eval_return"] = eval_reward

        wandb.log(log_dict, step=global_step)

        pbar.set_description(
            f"upd {update}/{args.num_iterations} | "
            f"SPS {sps:,.0f} | "
            f"ret {mr:.2f} | "
            f"len {ml:.0f} | "
            f"pg {pg_loss_val:.4f} | "
            f"vl {v_loss_val:.4f} | "
            f"ent {ent_loss_val:.3f} | "
            f"kl {approx_kl_val:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.1e}"
            f"{'  ⚠KL-STOP' if kl_early_stopped else ''}"
        )

        # ── checkpoint ───────────────────────────────────────────
        if update % args.save_interval == 0:
            save_checkpoint(
                args, agent, optimizer, obs_rms, ret_rms, global_step, update
            )

    # ── final save ───────────────────────────────────────────────
    save_checkpoint(
        args,
        agent,
        optimizer,
        obs_rms,
        ret_rms,
        global_step,
        args.num_iterations,
        name="agent_final.pt",
    )
    total_time = time.time() - t0
    print(f"\n✅  Training complete — {global_step:,} total steps in {total_time:.1f}s")
    print(f"Average speed: {global_step / total_time:,.0f} SPS")

    env.close()
    wandb.finish()
