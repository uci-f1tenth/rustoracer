from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from numpy.typing import NDArray
from rustoracerpy.rustoracer import PySim
import rerun as rr


class RustoracerEnv(gym.vector.VectorEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        yaml: str,
        num_envs: int = 1,
        max_steps: int = 10_000,
        render_mode: str | None = None,
    ):
        self._sim = PySim(yaml, num_envs, max_steps)
        nb, od = self._sim.n_beams, self._sim.obs_dim
        lo, hi = np.full(od, -100.0), np.full(od, 100.0)
        lo[:nb], hi[:nb] = self._sim.min_range, self._sim.max_range
        lo[nb : nb + 3], hi[nb : nb + 3] = [-5.0, -0.5, -10.0], [20.0, 0.5, 10.0]
        lo[nb + 3], hi[nb + 3] = -np.pi, np.pi
        lo[nb + 4], hi[nb + 4] = -5.0, 5.0
        lo[nb + 5 :], hi[nb + 5 :] = -20.0, 20.0

        single_obs = spaces.Box(lo, hi, dtype=np.float64)
        single_act = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64)
        self.num_envs = num_envs
        self.render_mode = render_mode
        self.single_observation_space = single_obs
        self.single_action_space = single_act
        self.observation_space = batch_space(single_obs, num_envs)
        self.action_space = batch_space(single_act, num_envs)
        self.skeleton: NDArray[np.float64] = self._sim.skeleton
        if render_mode == "human":
            rr.init("simple_image_display", spawn=True)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._sim.seed(seed)
        scans, _, _, _, states = self._sim.reset()
        return scans.reshape(self.num_envs, -1), {
            "state": states.reshape(self.num_envs, -1)
        }

    def step(self, actions):
        scans, rewards, term, trunc, states = self._sim.step(actions.ravel())
        return (
            scans.reshape(self.num_envs, -1),
            rewards,
            term,
            trunc,
            {"state": states.reshape(self.num_envs, -1)},
        )

    def easy_step(self, actions):
        scans, rewards, term, trunc, states = self._sim.easy_step(actions.ravel())
        return (
            scans.reshape(self.num_envs, -1),
            rewards,
            term,
            trunc,
            {"state": states.reshape(self.num_envs, -1)},
        )

    def render(self, step: int | None = None):
        """Render the current simulation state.

        Args:
            step: Optional step index used to set the rerun timeline sequence.
                  When provided, frames are ordered correctly in the rerun viewer.
        """
        if self.render_mode == "human":
            if step is not None:
                rr.set_time_sequence("step", step)
            rr.log("world/image", rr.Image(self._sim.render()))
            scans, _, _, _, state = self._sim.observe()
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
            rr.log(
                "world/lidar",
                rr.LineStrips2D(
                    np.stack(
                        [
                            self._sim.world_to_pixels(org.ravel()).reshape(-1, 2),
                            self._sim.world_to_pixels(ends.ravel()).reshape(-1, 2),
                        ],
                        axis=1,
                    )
                ),
            )
            rr.log(
                "world/car",
                rr.Points2D(
                    self._sim.car_pixels().reshape(-1, 2),
                    colors=[[43, 127, 255]],
                    radii=[1.0],
                ),
            )
        elif self.render_mode == "rgb_array":
            return self._sim.render()
