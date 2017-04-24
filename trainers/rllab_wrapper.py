from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.vpg import VPG

from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.base import Env as RllEnv
from rllab.envs.gym_env import GymEnv
from sandbox.rocky.tf.policies.base import StochasticPolicy as RllPolicy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.base import Baseline as RllBaseline
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.spaces import Box, Discrete, Product
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
from rllab.misc.instrument import run_experiment_lite
from rllab.core.serializable import Serializable

from models.repr import ModelState
from worlds.gym import GymWorld

from collections import OrderedDict
import numpy as np
import tensorflow as tf

import sys
import os

class RllEnvWrapper(RllEnv):
    def __init__(self, underlying):
        self.underlying = underlying
        self.active_instance = None
        super(RllEnvWrapper, self).__init__()
        self.reset()

    @property
    def action_space(self):
        if self.underlying.is_discrete:
            return Discrete(self.underlying.n_act)
        else:
            bound = np.asarray([1] * self.underlying.n_act)
            return Box(-bound, bound)

    @property
    def observation_space(self):
        bound = np.asarray([np.inf] * self.underlying.n_obs)
        return Box(-bound, bound)

    def reset(self):
        self.active_instance = self.underlying.sample_train()
        obs = self.underlying.reset([self.active_instance])[0]
        return obs

    def step(self, action):
        #print "before", self.active_instance, self.active_instance.state.blocks
        #print >>sys.stderr, action,
        assert self.active_instance is not None
        features, rewards, stops = self.underlying.step([action], [self.active_instance])
        #print "after", self.active_instance, self.active_instance.state.blocks
        #print
        #obs = (self.active_instance.task.id, features[0])
        obs = features[0]
        return obs, rewards[0], stops[0], {}

    def horizon(self):
        pass

class RllPolicyWrapper(RllPolicy, Serializable):
    def __init__(self, underlying, env_spec, env):
        self._Serializable__args = ()
        self._Serializable__kwargs = ()
        setattr(self, "_serializable_initialized", True)

        self.underlying = underlying
        self.env_spec = env_spec
        if isinstance(env_spec.action_space, Discrete):
            self.dist = Categorical(env_spec.action_space.n)
            self.is_discrete = True
        else:
            self.dist = DiagonalGaussian(env_spec.action_space.flat_dim)
            self.is_discrete = False
        self.mstate = None
        self.env = env
        super(RllPolicy, self).__init__(env_spec)

    @property
    def distribution(self):
        return self.dist

    @property
    def vectorized(self):
        return True

    def dist_info_sym(self, obs_var, state_info_vars):
        task_id_var = tf.cast(state_info_vars["task_id"], tf.int32)
        t_param, t_temp = self.underlying.prepare_sym(obs_var, task_id_var)
        if self.is_discrete:
            return {"prob": tf.nn.softmax(t_mean)}
        else:
            return {"mean": t_param, "log_std": t_temp}

    def reset(self, dones=None):
        self.initialized_mstates = False
        self.mstates = None

    def get_action(self, obs):
        if not self.initialized_mstates:
            assert self.env.active_instance is not None
            self.mstates = [ModelState(self.env.active_instance.task.id, (0,))]
            self.initialiezd_mstates = True

        actions, params = self.get_actions([obs])
        action = actions[0, :]
        return action, params

    def get_actions(self, obs):
        if not self.initialized_mstates:
            assert self.env.active_instance is not None
            self.mstates = [
                    ModelState(self.env.active_instance.task.id, (0,))
                    for _ in range(len(obs))]
            self.initialized_mstates = True

        session = tf.get_default_session()
        action_p, action_t, self.mstates = self.underlying.get_action(np.asarray(obs), self.mstates, None, session)

        if self.is_discrete:
            action_p = np.exp(action_p)
            action_p /= np.sum(action_p, axis=1)
            actions = [self.action_space.weighted_sample(p) for p in action_p]
            #action = self.action_space.weighted_sample(action_p)
            return actions, {
                "prob": action_p,
                "task_id": [m.task_id for m in self.mstates]
            }
        else:
            actions = [
                    self.dist.sample({"mean": m, "log_std": action_t})
                    for m in action_p]
            #action = self.dist.sample({"mean": action_p, "log_std": action_t})
            return actions, {
                "mean": action_p, "log_std": [action_t for _ in range(len(obs))],
                "task_id": [m.task_id for m in self.mstates]
            }

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

    def train(self, world, model, objective, eval_thunk=None):
        #if isinstance(world, GymWorld):
        #    env = TfEnv(GymEnv(world.env_name))
        #else:
        #    env = TfEnv(RllEnvWrapper(world))

        env = TfEnv(RllEnvWrapper(world))
        #env = TfEnv(GymEnv(world.env_name))

        policy = RllPolicyWrapper(model, env.spec, env._wrapped_env)
        #policy = CategoricalMLPPolicy("policy", env.spec)
        #policy = GaussianMLPPolicy("policy", env.spec)

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
            step_size=self.config.objective.step_size,
            entropy_bonus=self.config.objective.entropy_bonus,
            #n_parallel=10,
        )

        algo.train()
