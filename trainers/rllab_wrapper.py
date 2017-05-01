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
import rllab.misc.logger as rll_logger

from misc import util
from models.repr import ModelState
from worlds.gym import GymWorld

from collections import OrderedDict
import numpy as np
import tensorflow as tf

import sys
import os

class RllEnvWrapper(RllEnv):
    def __init__(self, underlying, guide):
        self.underlying = underlying
        self.active_instance = None
        self.max_hint_len = guide.max_len
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
        return Product(
                Box(-bound, bound),
                Box(np.asarray([0]), np.asarray([self.underlying.n_tasks])),
                Box(
                    np.asarray([0] * self.max_hint_len),
                    np.asarray([self.underlying.n_vocab] * self.max_hint_len)))

    def reset(self):
        self.active_instance = self.underlying.sample_train()
        underlying_obs, = self.underlying.reset([self.active_instance])
        padded_hint = (
                self.active_instance.task.hint 
                + (0,) * (self.max_hint_len - len(self.active_instance.task.hint)))
        obs = (underlying_obs, [self.active_instance.task.id], padded_hint)
        return obs

    def step(self, action):
        assert self.active_instance is not None
        features, rewards, stops = self.underlying.step([action], [self.active_instance])
        padded_hint = (
                self.active_instance.task.hint 
                + (0,) * (self.max_hint_len - len(self.active_instance.task.hint)))
        obs = (
                features[0], 
                [self.active_instance.task.id],
                padded_hint)
        return obs, rewards[0], stops[0], {}

    def horizon(self):
        pass

class RllPolicyWrapper(RllPolicy, Serializable):
    def __init__(self, underlying, env_spec, env, session):
        self._Serializable__args = ()
        self._Serializable__kwargs = ()
        setattr(self, "_serializable_initialized", True)

        self.underlying = underlying
        self.env_spec = env_spec
        if env.underlying.is_discrete:
            self.dist = Categorical(env_spec.action_space.n)
            self.is_discrete = True
        else:
            self.dist = DiagonalGaussian(env_spec.action_space.flat_dim)
            self.is_discrete = False
        self.env = env
        self.initialized_mstates = None
        self.mstates = None
        self.session = session
        super(RllPolicy, self).__init__(env_spec)

    @property
    def distribution(self):
        return self.dist

    @property
    def vectorized(self):
        return True

    def dist_info_sym(self, obs_var, state_info_vars):
        # TODO from spec
        n_obs = self.env.underlying.n_obs
        true_obs_var = obs_var[:, :n_obs]
        task_id_var = tf.cast(obs_var[:, n_obs], tf.int32)
        hint_var = tf.cast(obs_var[:, n_obs+1:], tf.int32)
        t_param, t_temp = self.underlying.prepare_sym(
                true_obs_var, task_id_var, hint_var)
        if self.is_discrete:
            return {"prob": tf.nn.softmax(t_param)}
        else:
            return {"mean": t_param, "log_std": t_temp}

    def reset(self, dones=None):
        if self.initialized_mstates is None:
            self.initialized_mstates = [False] * len(dones)
            self.mstates = [None] * len(dones)
        assert len(self.initialized_mstates) == len(dones)
        for i, d in enumerate(dones):
            if d:
                self.initialized_mstates[i] = False

    def get_action(self, obs):
        actions, params = self.get_actions([obs])
        action = actions[0]
        param = {k: v[0] for k, v in params.items()}
        return action, param

    def get_actions(self, obs):
        true_obs, task_ids, hints = zip(*obs)
        task_ids = np.asarray(task_ids).ravel()

        for i, im in enumerate(self.initialized_mstates):
            if not im:
                self.mstates[i] = ModelState(task_ids[i], hints[i])
                self.initialized_mstates[i] = True

        session = self.session
        action_p, action_t, self.mstates = self.underlying.get_action(
                np.asarray(true_obs), self.mstates, None, session)

        if self.is_discrete:
            action_p = np.exp(action_p)
            action_p /= np.sum(action_p, axis=1)[:, np.newaxis]
            actions = [self.action_space.weighted_sample(p) for p in action_p]
            return actions, {
                "prob": action_p,
            }
        else:
            actions = [
                    self.dist.sample({"mean": m, "log_std": action_t})
                    for m in action_p]
            return actions, {
                "mean": action_p, "log_std": [action_t for _ in range(len(obs))],
            }

    def get_params_internal(self, **tags):
        return self.underlying.params

class RllBaselineWrapper(RllBaseline):
    def __init__(self, underlying):
        self.underlying = underlying
        super(RllBaseline, self).__init__()

class RlLabTrainer(object):
    def __init__(self, config, world, model, guide, session):
        self.config = config
        self.session = session
        env = TfEnv(RllEnvWrapper(world, guide))
        policy = RllPolicyWrapper(model, env.spec, env._wrapped_env, session)
        baseline = LinearFeatureBaseline(env.spec)
        algo_ctor = globals()[self.config.trainer.algo]
        self.algo = algo_ctor(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=self.config.objective.n_train_batch,
            max_path_length=self.config.trainer.max_rollout_len,
            #n_itr=self.config.trainer.n_iters,
            discount=self.config.objective.discount,
            step_size=self.config.objective.step_size,
            entropy_bonus=self.config.objective.entropy_bonus,
            #n_parallel=10,
        )

    def train(self, world, model, objective, eval_thunk=None):
        self.session.run(tf.global_variables_initializer())
        self.algo.start_worker()
        i_iter = 0
        while True:
            with rll_logger.prefix("itr #%d | " % i_iter):
                paths = self.algo.obtain_samples(i_iter)
                samples_data = self.algo.process_samples(i_iter, paths)
                self.algo.log_diagnostics(paths)
                self.algo.optimize_policy(i_iter, samples_data)
                rll_logger.dump_tabular(with_prefix=True)
            i_iter += 1

            if i_iter % 100 == 0 and eval_thunk is not None:
                model.save(self.session)
                eval_thunk()

        self.algo.shutdown_worker()
