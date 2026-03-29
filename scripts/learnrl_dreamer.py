# scripts/learnrl_dreamerv3.py
from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Tuple
from collections import deque
import pathlib

import numpy as np
import torch
import torch.nn as nn
import tqdm
import tyro
import wandb
import ruamel.yaml

# Your Rust environment
from rustoracerpy import RustoracerEnv

# Import the prepped Dreamer backend files
import sys
sys.path.append(str(pathlib.Path(__file__).parent  / "dreamer" / "src"))
print(f"Using Dreamer code from: {sys.path[-1]}")
import tools
import models

print("-"*50)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device:      {torch.cuda.get_device_name(0)}")
print("-"*50)

if not torch.cuda.is_available():
    print("FATAL ERROR: PyTorch installed without CUDA support.")
    print("uv likely pulled the CPU wheel from standard PyPI instead of the PyTorch index.")
    sys.exit(1)

# ━━━━━━━━━━━━━━━━━━━━  hyper-parameters  ━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass 
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 42
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = True

    # Environment
    yaml_map: str = "maps/berlin.yaml" 
    total_timesteps: int = 10_000_000
    num_envs: int = 1024
    max_ep_steps: int = 10_000
    
    # Dreamer Data Collection
    train_every_steps: int = 128       # Train every N env steps (per env)
    train_updates: int = 4             # How many gradient updates to do per train phase
    prefill_steps: int = 2500          # Total random steps to take before training
    
    # Dreamer Buffer
    batch_size: int = 50               # Number of sequences per batch
    batch_length: int = 50             # Length of each sequence (Time)
    buffer_capacity: int = 100_000     # Max steps to store in RAM (per env)
    
    # Logging
    video_interval: int = 25
    video_max_steps: int = 500

# ━━━━━━━━━━━━━━━━━━━━  Vectorized Buffer  ━━━━━━━━━━━━━━━━━━━━━━━
class VectorizedReplayBuffer:
    def __init__(self, num_envs, max_steps, obs_dim, act_dim, device):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.device = device
        self.ptr = 0
        self.full = False

        self.obs = torch.zeros((max_steps, num_envs, obs_dim), dtype=torch.float32, device=device)
        self.action = torch.zeros((max_steps, num_envs, act_dim), dtype=torch.float32, device=device)
        self.reward = torch.zeros((max_steps, num_envs), dtype=torch.float32, device=device)
        self.discount = torch.ones((max_steps, num_envs), dtype=torch.float32, device=device)
        self.is_first = torch.zeros((max_steps, num_envs), dtype=torch.bool, device=device)
        self.is_terminal = torch.zeros((max_steps, num_envs), dtype=torch.bool, device=device)

    def add(self, obs, action, reward, done, is_first):
        def to_tensor(x, dtype=torch.float32):
            if not isinstance(x, torch.Tensor):
                return torch.tensor(x, dtype=dtype, device=self.device)
            return x.to(self.device, dtype=dtype)

        self.obs[self.ptr] = to_tensor(obs)
        self.action[self.ptr] = to_tensor(action)
        self.reward[self.ptr] = to_tensor(reward)
        self.discount[self.ptr] = 1.0 - to_tensor(done)
        self.is_first[self.ptr] = to_tensor(is_first, dtype=torch.bool)
        self.is_terminal[self.ptr] = to_tensor(done, dtype=torch.bool)

        self.ptr = (self.ptr + 1) % self.max_steps
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size, sequence_length):
        max_idx = self.max_steps if self.full else self.ptr
        if max_idx < sequence_length:
            raise ValueError("Not enough data in buffer to sample sequence.")

        start_idxs = torch.randint(0, max_idx - sequence_length + 1, (batch_size,))
        env_idxs = torch.randint(0, self.num_envs, (batch_size,))

        # The "Dummy Space" adapter: We map the flat array to the "lidar" key
        return {
            "lidar": torch.stack([self.obs[start:start+sequence_length, env] for start, env in zip(start_idxs, env_idxs)], dim=1),
            "action": torch.stack([self.action[start:start+sequence_length, env] for start, env in zip(start_idxs, env_idxs)], dim=1),
            "reward": torch.stack([self.reward[start:start+sequence_length, env] for start, env in zip(start_idxs, env_idxs)], dim=1),
            "discount": torch.stack([self.discount[start:start+sequence_length, env] for start, env in zip(start_idxs, env_idxs)], dim=1),
            "is_first": torch.stack([self.is_first[start:start+sequence_length, env] for start, env in zip(start_idxs, env_idxs)], dim=1),
            "is_terminal": torch.stack([self.is_terminal[start:start+sequence_length, env] for start, env in zip(start_idxs, env_idxs)], dim=1),
        }

# ━━━━━━━━━━━━━━━━━━━━  Video Evaluation  ━━━━━━━━━━━━━━━━━━━━━━━━
@torch.no_grad()
def record_eval_video(yaml_map: str, wm: models.WorldModel, behavior: models.ImagBehavior, device: torch.device, max_steps: int):
    eval_env = RustoracerEnv(yaml=yaml_map, num_envs=1, max_steps=10_000, render_mode="rgb_array")
    raw_obs, _ = eval_env.reset(seed=42)
    
    agent_state = None
    is_first = torch.ones(1, dtype=torch.bool, device=device)
    
    frames, total_reward = [], 0.0
    for _ in range(max_steps):
        frame = eval_env.render()
        if frame is not None: frames.append(frame)
        
        obs_t = torch.tensor(raw_obs, device=device, dtype=torch.float32)
        agent_obs = wm.preprocess({"lidar": obs_t, "is_first": is_first, "is_terminal": torch.zeros_like(is_first)})
        
        embed = wm.encoder(agent_obs)
        agent_state, _ = wm.dynamics.obs_step(agent_state, None if is_first[0] else action, embed, agent_obs["is_first"])
        
        feat = wm.dynamics.get_feat(agent_state)
        action = behavior.actor(feat).mode() 
        
        raw_obs, reward, term, trunc, _ = eval_env.step(action.cpu().numpy().astype(np.float64).clip(-1.0, 1.0))
        total_reward += float(reward[0])
        is_first = torch.tensor(term | trunc, dtype=torch.bool, device=device)
        
        if term[0] or trunc[0]: break
        
    eval_env.close()
    if not frames: return None, total_reward
    video_np = np.stack(frames).transpose(0, 3, 1, 2)
    return wandb.Video(video_np, fps=60, format="mp4"), total_reward

def _video_thread_fn(yaml_map, wm, behavior, device, max_steps, update, global_step):
    try:
        vid, eval_reward = record_eval_video(yaml_map, wm, behavior, device, max_steps)
        log_dict = {"charts/eval_return": eval_reward}
        if vid is not None: log_dict["media/eval_video"] = vid
        wandb.log(log_dict, step=global_step)
    except Exception as e:
        print(f"[iter {update}] Video thread error: {e}")

# ━━━━━━━━━━━━━━━━━━━━  Main Loop  ━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"DreamerV3_{args.exp_name}_{args.seed}"
    wandb.init(project="dreamerv3_rustoracer", name=run_name, config=vars(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    env = RustoracerEnv(yaml=args.yaml_map, num_envs=args.num_envs, max_steps=args.max_ep_steps)
    obs_dim, act_dim = env.single_observation_space.shape[0], env.single_action_space.shape[0]
    
    # --- The Config Strategy ---
    config_path = pathlib.Path(__file__).parent / "dreamer" / "src" / "configs.yaml" 
    yaml_parser = ruamel.yaml.YAML(typ="safe")
    config_dict = yaml_parser.load(config_path.read_text())["defaults"]
    config_dict["device"] = str(device)
    config_dict["num_actions"] = act_dim
    class ConfigWrapper:
        def __init__(self, d): self.__dict__.update(d)
    d_config = ConfigWrapper(config_dict)

    # --- Initialize Dreamer Networks (The Dummy Space Trick) ---
    class DummySpace: shape = (obs_dim,)
    class DummyObsSpace: spaces = {"lidar": DummySpace()}
    
    wm = models.WorldModel(DummyObsSpace(), act_dim, 0, d_config).to(device)
    behavior = models.ImagBehavior(d_config, wm).to(device)
    
    actual_buffer_capacity = max(args.batch_length * 2, args.buffer_capacity // args.num_envs)
    buffer = VectorizedReplayBuffer(args.num_envs, actual_buffer_capacity, obs_dim, act_dim, device)

    # --- State Tracking ---
    next_obs_np, _ = env.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    is_first = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    agent_state, last_action = None, None
    
    ep_ret_cpu = np.zeros(args.num_envs, dtype=np.float32)
    ep_len_cpu = np.zeros(args.num_envs, dtype=np.int32)
    recent_returns, recent_lengths = deque(maxlen=200), deque(maxlen=200)

    # --- Prefill Phase ---
    prefill_iters = max(1, args.prefill_steps // args.num_envs)
    print(f"Prefilling buffer with random actions ({prefill_iters} steps per env)...")
    for _ in range(prefill_iters):
        action_np = np.random.uniform(-1.0, 1.0, size=(args.num_envs, act_dim)).astype(np.float64)
        obs_np, reward_np, term_np, trunc_np, _ = env.step(action_np)
        done_np = term_np | trunc_np
        buffer.add(next_obs_np, action_np, reward_np, done_np, is_first.cpu().numpy())
        next_obs_np, is_first = obs_np, torch.tensor(done_np, dtype=torch.bool, device=device)

    # --- Training Phase ---
    global_step = 0
    num_iterations = args.total_timesteps // (args.train_every_steps * args.num_envs)
    pbar = tqdm.tqdm(range(1, num_iterations + 1))
    t0 = time.time()
    video_thread = None

    for update in pbar:
        # 1. Rollout Collection
        for step in range(args.train_every_steps):
            with torch.inference_mode():
                agent_obs = wm.preprocess({"lidar": next_obs, "is_first": is_first, "is_terminal": torch.zeros_like(is_first)})
                embed = wm.encoder(agent_obs)
                agent_state, _ = wm.dynamics.obs_step(agent_state, last_action, embed, agent_obs["is_first"])
                feat = wm.dynamics.get_feat(agent_state)
                action = behavior.actor(feat).sample()
                last_action = action

            action_np = action.cpu().numpy().astype(np.float64).clip(-1.0, 1.0)
            obs_np, reward_np, term_np, trunc_np, _ = env.step(action_np)
            done_np = term_np | trunc_np

            ep_ret_cpu += reward_np
            ep_len_cpu += 1
            for i in np.where(done_np)[0]:
                recent_returns.append(ep_ret_cpu[i])
                recent_lengths.append(ep_len_cpu[i])
                ep_ret_cpu[i] = 0.0; ep_len_cpu[i] = 0

            buffer.add(next_obs_np, action_np, reward_np, done_np, is_first.cpu().numpy())
            next_obs_np = obs_np
            next_obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
            is_first = torch.tensor(done_np, dtype=torch.bool, device=device)
            global_step += args.num_envs

        # 2. Dreamer Update (Imagination)
        for _ in range(args.train_updates):
            batch = buffer.sample(args.batch_size, args.batch_length)
            post, context, wm_metrics = wm._train(batch)
            def reward_fn(f, s, a): return wm.heads["reward"](wm.dynamics.get_feat(s)).mode()
            imag_feat, imag_state, imag_action, weights, ac_metrics = behavior._train(post, reward_fn)

        # 3. Logging & Video
        sps = global_step / max(time.time() - t0, 1e-6)
        mr = float(np.mean(recent_returns)) if recent_returns else 0.0
        ml = float(np.mean(recent_lengths)) if recent_lengths else 0.0
        
        log_dict = {"charts/episode_return": mr, "charts/episode_length": ml, "charts/SPS": sps}
        for k, v in {**wm_metrics, **ac_metrics}.items():
            if v is not None:
                if hasattr(v, 'detach'): 
                    v = v.detach().cpu().numpy()
                log_dict[f"losses/{k}"] = float(np.mean(v))
        wandb.log(log_dict, step=global_step)

        if update % args.video_interval == 0 and args.capture_video:
            if video_thread and video_thread.is_alive(): video_thread.join()
            wm_snap, beh_snap = type(wm)(DummyObsSpace(), act_dim, 0, d_config).to(device), type(behavior)(d_config, wm).to(device)
            wm_snap.load_state_dict(wm.state_dict()); beh_snap.load_state_dict(behavior.state_dict())
            video_thread = threading.Thread(target=_video_thread_fn, args=(args.yaml_map, wm_snap, beh_snap, device, args.video_max_steps, update, global_step), daemon=True)
            video_thread.start()

        pbar.set_description(f"upd {update}/{num_iterations} | SPS {sps:,.0f} | ret {mr:.2f} | len {ml:.0f} | WM Loss {wm_metrics.get('model_loss', 0):.3f}")

    if video_thread and video_thread.is_alive(): video_thread.join()
    env.close()
    wandb.finish()