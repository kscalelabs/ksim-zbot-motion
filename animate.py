#!/usr/bin/env python
"""Hard-coded, one-off: load a Mujoco animation → feed it as actions each ctrl_dt."""

import asyncio
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import ksim
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray
from mujoco_animator import MjAnim

from train import (
    ZEROS,
    Model,
    ZbotWalkingTask,
    ZbotWalkingTaskConfig,
)

MJCF_PATH = asyncio.run(ksim.get_mujoco_model_path("zbot", name="robot"))
ANIM_PATHS = [Path("gaits/stand.json"), Path("gaits/stand_to_step.json")]
LOOP = False
INTERP: Literal["linear", "cubic"] = "cubic"


def load_trajectory(dt: float) -> jnp.ndarray:
    clips = [MjAnim.load(p).to_numpy(dt, interp=INTERP, loop=False) for p in ANIM_PATHS]
    return jnp.concatenate(clips, axis=0)


class ZbotAnimateTask(ZbotWalkingTask):
    """Task that plays back animation clips instead of using a trained model."""

    def __init__(self, config: ZbotWalkingTaskConfig) -> None:
        super().__init__(config)
        self.trajectory = load_trajectory(config.ctrl_dt)
        self.T = self.trajectory.shape[0]
        self.zeros_j = jnp.array([angle for _, angle in ZEROS])

    def sample_action(
        self,
        model: Model,
        model_carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        if physics_state.data.time < 1e-9:
            physics_state.data.qpos[7:] = self.zeros_j
            physics_state.data.qvel[:] = 0.0  # start at rest
            mujoco.mj_forward(physics_model, physics_state.data)

        step = int(jnp.round(physics_state.data.time / self.config.ctrl_dt))
        frame = self.trajectory[step % self.T] if LOOP else self.trajectory[min(step, self.T - 1)]
        joint_positions = frame[7:]
        return ksim.Action(action=joint_positions, carry=model_carry)


if __name__ == "__main__":
    cfg = ZbotWalkingTaskConfig(
        run_mode="view",
        rollout_length_seconds=999,
        num_envs=1,
        batch_size=1,
        dt=0.001,
        ctrl_dt=0.02,
        iterations=8,
        ls_iterations=8,
        render_shadow=False,
        render_reflection=False,
        live_reward_buffer_size=4,
    )
    task = ZbotAnimateTask(cfg)
    task.run_model_viewer(argmax_action=True)
