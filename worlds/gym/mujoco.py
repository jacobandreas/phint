import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six

from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.misc import logger

from gym import utils
from gym.wrappers import Monitor

MODEL_TEMPLATE = "/home/ubuntu/phint/worlds/gym/maps/map_%s.xml"

try:
    import mujoco_py
    from mujoco_py.mjlib import mjlib
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

class DynamicMujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, task_names, frame_skip):
        self.frame_skip = frame_skip

        #self.models = [mujoco_py.MjModel("/home/ubuntu/phint/worlds/gym/maps/map_%s.xml" % t) for t in task_names]
        self.task_names = task_names
        self.models = [None] * len(task_names)

        self.task_datas = []
        for t in task_names:
            task_data = []
            with open("/home/ubuntu/phint/worlds/gym/maps/data_%s.txt" % t) as data_f:
                lines = data_f.readlines()
                for line in lines[:4]:
                    task_data.append(tuple(float(d) for d in line.strip().split()))
                task_data.append(lines[4].strip().split())
            self.task_datas.append(task_data)
        self.viewer = None
        self.next_task = None
        self._load_map(init=True)

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        #self.init_qpos = self.model.data.qpos.ravel().copy()
        #self.init_qvel = self.model.data.qvel.ravel().copy()
        observation, _reward, done, _info = self._step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _load_map(self, init=False):
        if self.next_task is None:
            if not init:
                logger.log("warning: reset with empty map")
            self.i_active = 0
        else:
            self.i_active = self.next_task
            self.next_task = None

        #self.i_active = np.random.randint(len(self.models))
        if self.models[self.i_active] is None:
            self.models[self.i_active] = mujoco_py.MjModel(
                    "/home/ubuntu/phint/worlds/gym/maps/map_%s.xml" % 
                        self.task_names[self.i_active])
        self.model = self.models[self.i_active]
        self.task_data = self.task_datas[self.i_active]
        self.data = self.model.data
        self.init_qpos = self.model.data.qpos.ravel().copy()
        self.init_qvel = self.model.data.qvel.ravel().copy()
        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer_setup()

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def _reset(self):
        mjlib.mj_resetData(self.model.ptr, self.data.ptr)
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer.autoscale()
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model._compute_subtree()  # pylint: disable=W0212
        self.model.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.model.data.ctrl = ctrl
        for _ in range(n_frames):
            self.model.step()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self._get_viewer().finish()
                self.viewer = None
            return

        if mode == 'rgb_array':
            self._get_viewer().render()
            data, width, height = self._get_viewer().get_image()
            return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().loop_once()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer()
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(six.b(body_name))
        return self.model.data.com_subtree[idx]

    def get_body_comvel(self, body_name):
        idx = self.model.body_names.index(six.b(body_name))
        return self.model.body_comvels[idx]

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(six.b(body_name))
        return self.model.data.xmat[idx].reshape((3, 3))

    def state_vector(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat
        ])

class AntEnv(DynamicMujocoEnv, utils.EzPickle):
    def __init__(self):
        DynamicMujocoEnv.__init__(self, [str(i) for i in range(50)], 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        self._load_map()
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

def convert_gym_space(space):
    if isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    else:
        raise NotImplementedError


class CappedCubicVideoSchedule(object):
    def __call__(self, count):
        if count < 1000:
            return int(round(count ** (1. / 3))) ** 3 == count
        else:
            return count % 1000 == 0

class FixedIntervalVideoSchedule(object):

    def __init__(self, interval):
        self.interval = interval

    def __call__(self, count):
        return count % self.interval == 0


class NoVideoSchedule(object):
    def __call__(self, count):
        return False


class GymEnv(Env, Serializable):
    def __init__(self, wrapped_env, env_spec, record_video=True, video_schedule=None,
            log_dir=None, force_reset=False):
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
        Serializable.quick_init(self, locals())

        #env = gym.envs.make(env_name)
        env = wrapped_env
        self.env = env
        self.env_id = env_spec.id
        self.env.spec = env_spec

        #self.env_id = env_name
        #assert env_name == env.spec.id
        #self.env_id = env.spec.id

        if log_dir is None:
            self.monitoring = False
        else:
            if video_schedule is None:
                video_schedule = CappedCubicVideoSchedule()
            self.env = Monitor(self.env, log_dir, video_callable=video_schedule, force=True)
            self.monitoring = True

        ### monitor.logger.setLevel(logging.WARNING)
        ### if log_dir is None:
        ###     self.monitoring = False
        ### else:
        ###     if not record_video:
        ###         video_schedule = NoVideoSchedule()
        ###     else:
        ###         if video_schedule is None:
        ###             video_schedule = CappedCubicVideoSchedule()
        ###     self.env.monitor.start(log_dir, video_schedule)
        ###     self.monitoring = True

        self._observation_space = convert_gym_space(env.observation_space)
        self._action_space = convert_gym_space(env.action_space)
        self._horizon = env.spec.timestep_limit
        self._log_dir = log_dir
        self._force_reset = force_reset

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self, task_id):
        if self._force_reset and self.monitoring:
            from gym.wrappers.monitoring import Monitor
            assert isinstance(self.env, Monitor)
            recorder = self.env.stats_recorder
            if recorder is not None:
                recorder.done = True
        
        self.env.unwrapped.next_task = task_id
        return self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return Step(next_obs, reward, done, **info)

    def render(self):
        self.env.render()

    def terminate(self):
        if self.monitoring:
            self.env.monitor.close()
            if self._log_dir is not None:
                print("""
    ***************************

    Training finished! You can upload results to OpenAI Gym by running the following command:

    python scripts/submit_gym.py %s

    ***************************
                """ % self._log_dir)
