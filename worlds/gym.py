from misc import util

from rllab.envs.gym_env import GymEnv
from rllab.spaces import Box, Discrete, Product
import rllab.misc.logger as logger

from collections import namedtuple
import os

GymTask = namedtuple("GymTask", ["id"])

class GymWorld(object):
    def __init__(self, config):
        rllab_dir = os.path.join(config.experiment_dir, "rllab")
        os.mkdir(rllab_dir)
        #logger.set_snapshot_dir(rllab_dir)

        self.env_name = config.world.env
        self.env = GymEnv(self.env_name)
        spec = self.env.spec

        self.max_hint_len = 1
        self.vocab = util.Index()
        self.task_index = util.Index()
        self.n_tasks = 1

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
        return GymInstance(GymTask(0), None, None)

    def reset(self, insts):
        assert len(insts) == 1
        inst, = insts
        assert inst._reset_id is None
        self._reset_counter[0] += 1
        inst._reset_id = self._reset_counter[0]
        obs = self.env.reset()
        return [obs]

    def step(self, actions, insts):
        assert len(actions) == 1
        action, = actions
        inst, = insts
        assert inst._reset_id == self._reset_counter[0], (inst._reset_id, self._reset_counter)
        step = self.env.step(action)
        return [step.observation], [step.reward], [step.done]

class GymInstance(object):
    def __init__(self, task, state, demo):
        self._reset_id = None

        self.task = task
        self.state = state
        self.demo = demo
