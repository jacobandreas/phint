from misc import util

from rllab.envs.gym_env import GymEnv
from rllab.envs.gym_env import gym as true_gym # TODO yikes
from rllab.spaces import Box, Discrete, Product
import rllab.misc.logger as logger

from collections import namedtuple
import os
import numpy as np

GymTask = namedtuple("GymTask", ["id"])

class GymWorld(object):
    def __init__(self, config):
        rllab_dir = os.path.join(config.experiment_dir, "rllab")
        os.mkdir(rllab_dir)
        logger.set_snapshot_dir(rllab_dir)

        self.env_name = config.world.env
        self.rll_env = GymEnv(self.env_name, force_reset=True)
        spec = self.rll_env.spec

        self.max_hint_len = 1
        self.vocab = util.Index()
        self.task_index = util.Index()
        self.n_tasks = 4
        self.random = util.next_random()

        self._reset_counter = [0]

        if isinstance(spec.action_space, Box):
            self.is_discrete = False
            self.n_act = spec.action_space.flat_dim
        else:
            self.is_discrete = True
            self.n_act = spec.action_space.flat_dim

        assert isinstance(spec.observation_space, Box)
        self.n_obs = spec.observation_space.flat_dim

    def sample_train(self, p=None):
        return GymInstance(GymTask(np.random.randint(self.n_tasks)), None, None)

    def reset(self, insts):
        assert len(insts) == 1
        inst, = insts
        assert inst._reset_id is None
        self._reset_counter[0] += 1
        inst._reset_id = self._reset_counter[0]
        obs = self.rll_env.reset()
        return [obs]

    def step(self, actions, insts):
        assert len(actions) == 1
        action, = actions
        inst, = insts
        assert inst._reset_id == self._reset_counter[0], (inst._reset_id, self._reset_counter)

        gym_env = self.rll_env.env.unwrapped

        x_before, y_before = gym_env.get_body_com("torso")[:2]
        #x_before, y_before = gym_env.model.data.qpos[:2, 0]
        step = self.rll_env.step(action)
        x_after, y_after = gym_env.get_body_com("torso")[:2]
        #x_after, y_after = gym_env.model.data.qpos[:2, 0]

        x_reward = (x_after - x_before) / gym_env.dt
        y_reward = (y_after - y_before) / gym_env.dt


        assert inst.task.id in (0, 1, 2, 3)
        if inst.task.id == 0:
            forward_reward = x_reward
        elif inst.task.id == 1:
            forward_reward = -x_reward
        elif inst.task.id == 2:
            forward_reward = y_reward
        elif inst.task.id == 3:
            forward_reward = -y_reward

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(gym_env.model.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        return [step.observation], [reward], [step.done]

class GymInstance(object):
    def __init__(self, task, state, demo):
        self._reset_id = None

        self.task = task
        self.state = state
        self.demo = demo
