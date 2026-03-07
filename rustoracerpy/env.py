from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from gymnasium.vector.utils import batch_space
from rustoracerpy.rustoracer import PySim

import rerun as rr


class RustoracerEnv(gym.vector.VectorEnv):
    metadata: dict[str, list[str] | int] = {
        "render_modes": ["rgb_array"],
        "render_fps": 60,
    }

    def __init__(
        self,
        yaml: str,
        num_envs: int = 1,
        max_steps: int = 10_000,
        render_mode: str | None = None,
    ) -> None:
        self._sim: PySim = PySim(yaml, num_envs, max_steps)

        single_obs_space = spaces.Box(
            np.array([self._sim.min_range] * self._sim.n_beams + [-5.0, -0.5, -10.0]),
            np.array([self._sim.max_range] * self._sim.n_beams + [20.0, 0.5, 10.0]),
            dtype=np.float64,
        )
        single_act_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64)

        self.num_envs = num_envs
        self.render_mode = render_mode
        self.single_observation_space = single_obs_space
        self.single_action_space = single_act_space
        self.observation_space = batch_space(single_obs_space, num_envs)
        self.action_space = batch_space(single_act_space, num_envs)

        self.skeleton: NDArray[np.float64] = self._sim.skeleton

        if render_mode == "human":
            rr.init("simple_image_display", spawn=True)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._sim.seed(seed)
        scans, _r, _t, _tr, states = self._sim.reset()
        return scans.reshape(self.num_envs, -1), {
            "state": states.reshape(self.num_envs, -1)
        }

    def step(self, actions):
        scans, rewards, terminated, truncated, states = self._sim.step(actions.ravel())
        return (
            scans.reshape(self.num_envs, -1),
            rewards,
            terminated,
            truncated,
            {"state": states.reshape(self.num_envs, -1)},
        )

    def easy_step(self, actions):
        scans, rewards, terminated, truncated, states = self._sim.easy_step(
            actions.ravel()
        )
        return (
            scans.reshape(self.num_envs, -1),
            rewards,
            terminated,
            truncated,
            {"state": states.reshape(self.num_envs, -1)},
        )

    def render(self):
        if self.render_mode == "human":
            img = self._sim.render()
            rr.log("world/image", rr.Image(img))

            scans, rewards, terminated, truncated, state = self._sim.observe()

            sk_px = self._sim.world_to_pixels(self.skeleton).reshape(-1, 2)
            rr.log("world/centerline", rr.LineStrips2D([sk_px]))

            angles = state[2] + np.linspace(
                -self._sim.fov / 2, self._sim.fov / 2, self._sim.n_beams
            )
            ends = np.c_[
                state[0] + scans[: self._sim.n_beams] * np.cos(angles),
                state[1] + scans[: self._sim.n_beams] * np.sin(angles),
            ]
            org = np.broadcast_to([[state[0], state[1]]], ends.shape)
            ends_px = self._sim.world_to_pixels(ends.ravel()).reshape(-1, 2)
            org_px = self._sim.world_to_pixels(org.ravel()).reshape(-1, 2)
            rr.log("world/lidar", rr.LineStrips2D(np.stack([org_px, ends_px], axis=1)))
            car_px = self._sim.car_pixels().reshape(-1, 2)
            rr.log(
                "world/car", rr.Points2D(car_px, colors=[[43, 127, 255]], radii=[1.0])
            )
        elif self.render_mode == "rgb_array":
            return self._sim.render()
