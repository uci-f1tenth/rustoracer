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
"""Rustoracer PPO agent → AutoDrive F1TENTH bridge."""

import time, numpy as np, torch, torch.nn as nn, socketio, eventlet
from flask import Flask
import autodrive


# ── Agent (must match training architecture) ──
class Agent(nn.Module):
    def __init__(self, obs=110, act=2, h=256):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, act)
        )

    def forward(self, x):
        return self.actor_mean(x)


# ── Load checkpoint ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(
    "checkpoints/agent_39321600.pt", map_location=DEVICE, weights_only=False
)
agent = Agent()
agent.load_state_dict(ckpt["model"], strict=False)
agent.to(DEVICE).eval()
obs_mean, obs_var = ckpt["obs_rms_mean"].numpy(), ckpt["obs_rms_var"].numpy()

# ── Helpers ──
LIDAR_IDX = np.linspace(0, 1080, 108, dtype=int)
prev_pos, prev_t, vel = None, None, 0.0


def get_obs(f):
    global prev_pos, prev_t, vel
    now = time.time()
    if prev_pos is not None and prev_t is not None:
        dt = now - prev_t
        if dt > 1e-6:
            vel = np.linalg.norm(f.position - prev_pos) / dt
    prev_pos, prev_t = f.position.copy(), now
    obs = np.concatenate([f.lidar_range_array[LIDAR_IDX], [vel, f.steering]])
    return np.clip((obs - obs_mean) / np.sqrt(obs_var + 1e-8), -10, 10)


# ── AutoDrive bridge ──
f1 = autodrive.F1TENTH()
f1.id = "V1"
sio = socketio.Server()
app = Flask(__name__)


@sio.on("connect")
def connect(sid, environ):
    print("Connected!")


@sio.on("Bridge")
def bridge(sid, data):
    if not data:
        return
    f1.parse_data(data)
    with torch.no_grad():
        act = agent(
            torch.tensor(get_obs(f1), device=DEVICE, dtype=torch.float32).unsqueeze(0)
        )
    a = act.cpu().numpy().flatten().clip(-1, 1)
    f1.steering_command, f1.throttle_command = float(a[0]), (float(a[1]) + 1) / 2
    try:
        sio.emit("Bridge", data=f1.generate_commands())
    except Exception as e:
        print(e)


if __name__ == "__main__":
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
