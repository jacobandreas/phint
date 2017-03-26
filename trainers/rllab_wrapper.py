from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.vpg import VPG

from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.base import Env as RllEnv
from sandbox.rocky.tf.policies.base import StochasticPolicy as RllPolicy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.base import Baseline as RllBaseline
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.spaces import Box, Discrete, Product
from sandbox.rocky.tf.distributions.categorical import Categorical

from rllab.misc.instrument import run_experiment_lite

from models.repr import ModelState

from collections import OrderedDict
import numpy as np
import tensorflow as tf

import sys


class RllEnvWrapper(RllEnv):
    def __init__(self, underlying):
        self.underlying = underlying
        self.active_instance = None
        super(RllEnvWrapper, self).__init__()

    @property
    def action_space(self):
        return Discrete(self.underlying.n_act)

    @property
    def observation_space(self):
        bound = np.asarray([np.inf] * self.underlying.n_obs)
        return Box(-bound, bound)

    def reset(self):
        #print >>sys.stderr, ""
        self.active_instance = self.underlying.sample_train()
        obs = self.underlying.reset([self.active_instance])[0]
        return obs

    def step(self, action):
        #print >>sys.stderr, action,
        assert self.active_instance is not None
        features, rewards, stops = self.underlying.step([action], [self.active_instance])
        #obs = (self.active_instance.task.id, features[0])
        obs = features[0]
        return obs, rewards[0], stops[0], {}

    def horizon(self):
        pass

class RllPolicyWrapper(RllPolicy):
    def __init__(self, underlying, env_spec, env):
        self.underlying = underlying
        self.env_spec = env_spec
        self.dist = Categorical(env_spec.action_space.n)
        self.mstate = None
        self.env = env
        super(RllPolicy, self).__init__(env_spec)

    @property
    def distribution(self):
        return self.dist

    def dist_info_sym(self, obs_var, state_info_vars):
        task_id_var = tf.cast(state_info_vars["task_id"], tf.int32)
        t_sym = self.underlying.prepare_sym(obs_var, task_id_var)
        return {
            "prob": t_sym,
        }

    def reset(self):
        self.mstate = ModelState(self.env.active_instance.task.id, (0,))

    def get_action(self, obs):
        #if obs is None:
        #    obs = np.zeros(self.env_spec.observation_space.flat_dim)

        session = tf.get_default_session()
        action_p = self.underlying.get_action_param(np.asarray([obs]), [self.mstate], None, session)
        assert action_p.shape[0] == 1
        action_p = action_p[0, :]
        action_p = np.exp(action_p)
        action_p /= np.sum(action_p)
        action = self.action_space.weighted_sample(action_p)
        return action, {"prob": action_p, "task_id": self.mstate.task_id}

    def get_params_internal(self, **tags):
        return self.underlying.params

    @property
    def state_info_specs(self):
        return [("task_id", ())]

class RllBaselineWrapper(RllBaseline):
    def __init__(self, underlying):
        self.underlying = underlying
        super(RllBaseline, self).__init__()

class RlLabTrainer(object):
    def __init__(self, config, _ignore_session):
        self.config = config

    def train(self, world, model, objective):
        env = TfEnv(RllEnvWrapper(world))
        policy = RllPolicyWrapper(model, env.spec, env._wrapped_env)
        #policy = CategoricalMLPPolicy("policy", env.spec)
        baseline = LinearFeatureBaseline(env.spec)
        #baseline=RllBaselineWrapper(model),

        algo_ctor = globals()[self.config.trainer.algo]
        algo = algo_ctor(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=self.config.objective.n_train_batch,
            max_path_length=self.config.trainer.max_rollout_len,
            n_itr=self.config.trainer.n_iters,
            discount=self.config.objective.discount,
            step_size=self.config.objective.step_size
        )

        algo.train()
