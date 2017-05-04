from misc import util

from mujoco import AntEnv, GymEnv

#from rllab.envs.gym_env import GymEnv
from rllab.envs.gym_env import gym as true_gym # TODO yikes
from rllab.spaces import Box, Discrete, Product
import rllab.misc.logger as logger

from collections import namedtuple
import os
import numpy as np

#GymTask = namedtuple("GymTask", ["id", "hint"])
class GymTask(object):
    def __init__(self, id, hint):
        self.id = id
        self.hint = hint

class GymWorld(object):
    def __init__(self, config):
        rllab_dir = os.path.join(config.experiment_dir, "rllab")
        os.mkdir(rllab_dir)
        logger.set_snapshot_dir(rllab_dir)

        #self.env_name = config.world.env
        #self.rll_env = GymEnv(self.env_name, force_reset=True)

        gym_env = AntEnv()
        gym_spec = true_gym.envs.spec("Ant-v1")
        #gym_env.spec = gym_spec
        self.rll_env = GymEnv(gym_env, gym_spec, force_reset=True)
        spec = self.rll_env.spec

        self.max_hint_len = 2
        self.vocab = util.Index()
        #self.task_index = util.Index()
        self.n_tasks = 50
        #self.n_tasks = 4 * 3
        #self.n_tasks = 2
        self.random = util.next_random()

        self.tasks = []
        #for i in range(self.n_tasks):
        #    task = GymTask(i, (self.vocab.index(str(i)),))
        #    self.task_index.index(task)
        #    self.tasks.append(task)

        self.i_red = self.vocab.index("red")
        self.i_green = self.vocab.index("green")
        self.i_blue = self.vocab.index("blue")
        self.i_north = self.vocab.index("north")
        self.i_east = self.vocab.index("east")
        self.i_south = self.vocab.index("south")
        self.i_west = self.vocab.index("west")

        #i_task = 0
        ##for color in ("red",):
        #for color in ("red", "green", "blue"):
        ##    for direction in ("east",):
        #    for direction in ("north", "east", "south", "west"):
        #        hint = (self.vocab[color], self.vocab[direction])
        #        task = GymTask(i_task, hint)
        #        #self.task_index.index(task)
        #        self.tasks.append(task)
        #        i_task += 1

        self.n_vocab = len(self.vocab)

        self._reset_counter = [0]

        if isinstance(spec.action_space, Box):
            self.is_discrete = False
            self.n_act = spec.action_space.flat_dim
        else:
            self.is_discrete = True
            self.n_act = spec.action_space.flat_dim

        assert isinstance(spec.observation_space, Box)
        self.n_obs = spec.observation_space.flat_dim + 6

    def sample_train(self, p=None):
        task_id = np.random.randint(40)
        return GymInstance(GymTask(task_id, None), None, None)

    def sample_val(self, p=None):
        task_id = np.random.randint(10) + 40
        return GymInstance(GymTask(task_id, None), None, None)

    def reset(self, insts):
        assert len(insts) == 1
        inst, = insts
        assert inst._reset_id is None
        self._reset_counter[0] += 1
        inst._reset_id = self._reset_counter[0]
        obs = self.rll_env.reset(inst.task.id)
        #return [obs]
        gym_env = self.rll_env.env.unwrapped
        inst.task.hint = (self.vocab[gym_env.task_data[4][0]], self.vocab[gym_env.task_data[4][1]])
        assert None not in inst.task.hint
        return [np.zeros(self.n_obs)]

    def step(self, actions, insts):
        assert len(actions) == 1
        action, = actions
        inst, = insts
        assert inst._reset_id == self._reset_counter[0], (inst._reset_id, self._reset_counter)

        gym_env = self.rll_env.env.unwrapped

        x_goal, y_goal = gym_env.task_data[3]
        #h_dir = inst.task.hint[1]
        #if h_dir == self.i_north:
        #    y_goal += 3
        #elif h_dir == self.i_east:
        #    x_goal += 3
        #elif h_dir == self.i_south:
        #    y_goal -= 3
        #elif h_dir == self.i_west:
        #    x_goal -= 3

        x_before, y_before = gym_env.get_body_com("torso")[:2]
        #x_before, y_before = gym_env.model.data.qpos[:2, 0]
        step = self.rll_env.step(action)
        x_after, y_after = gym_env.get_body_com("torso")[:2]
        #x_after, y_after = gym_env.model.data.qpos[:2, 0]

        #x_reward = (x_after - x_before) / gym_env.dt
        #y_reward = (y_after - y_before) / gym_env.dt

        #assert inst.task.id in (0, 1, 2, 3)
        #if inst.task.id == 0:
        #    forward_reward = x_reward
        #elif inst.task.id == 1:
        #    forward_reward = -x_reward
        #elif inst.task.id == 2:
        #    forward_reward = y_reward
        #elif inst.task.id == 3:
        #    forward_reward = -y_reward

        x_red, y_red = gym_env.task_data[0]
        x_green, y_green = gym_env.task_data[1]
        x_blue, y_blue = gym_env.task_data[2]
        x_to_red, y_to_red = x_after - x_red, y_after - y_red
        x_to_green, y_to_green = x_after - x_green, y_after - y_green
        x_to_blue, y_to_blue = x_after - x_blue, y_after - y_blue

        dist_before = np.abs(x_goal - x_before) + np.abs(y_goal - y_before)
        dist_after = np.abs(x_goal - x_after) + np.abs(y_goal - y_after)
        forward_reward = (dist_before - dist_after) / gym_env.dt

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(gym_env.model.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        obs = np.concatenate((
            step.observation,
            np.clip([x_to_red, y_to_red, x_to_blue, y_to_blue, x_to_green,
                y_to_green], -1, 1)))

        return [obs], [reward], [step.done]

class GymInstance(object):
    def __init__(self, task, state, demo):
        self._reset_id = None

        self.task = task
        self.state = state
        self.demo = demo
