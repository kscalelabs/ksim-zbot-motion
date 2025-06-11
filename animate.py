#!/usr/bin/env python
"""Hard-coded, one-off: load a Mujoco animation → feed it as actions each ctrl_dt."""

import asyncio
from pathlib import Path

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
INTERP = "cubic"


def load_trajectory(dt: float) -> jnp.ndarray:
    clips = [MjAnim.load(p).to_numpy(dt, interp=INTERP, loop=False) for p in ANIM_PATHS]
    return jnp.concatenate(clips, axis=0)


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
    task = ZbotWalkingTask(cfg)

    trajectory = load_trajectory(cfg.ctrl_dt)
    T = trajectory.shape[0]

    # Get the starting pose directly from train.py
    ZEROS_J = jnp.array([angle for _, angle in ZEROS])

    def constant_clip_sample_action(
        self: ZbotWalkingTask,
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
            physics_state.data.qpos[7:] = ZEROS_J
            physics_state.data.qvel[:] = 0.0  # start at rest
            mujoco.mj_forward(physics_model, physics_state.data)

        step = int(jnp.round(physics_state.data.time / cfg.ctrl_dt))
        frame = trajectory[step % T] if LOOP else trajectory[min(step, T - 1)]
        joint_positions = frame[7:]
        return ksim.Action(action=joint_positions, carry=model_carry)

    task.sample_action = constant_clip_sample_action.__get__(task, ZbotWalkingTask)
    task.run_model_viewer(argmax_action=True)
