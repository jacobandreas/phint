from trainers.curriculum import _do_rollout

import logging
import numpy as np
import tensorflow as tf

import models

#from sandbox.rocky.tf.envs.base import TfEnv
#from sandbox.rocky.tf.algos.trpo import TRPO
#from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
#from trainers.rllab_wrapper import RllEnvWrapper, RllPolicyWrapper
#import rllab.misc.logger as rll_logger

class AdaptationEvaluator(object):
    def __init__(self, config, world, model, objective, session):
        self.config = config
        self.world = world
        self.model = model
        self.objective = objective
        self.session = session

    def evaluate(self):
        logging.info("[ADAPTATION EVAL]")
        
        n_batch = self.config.trainer.n_rollout_batch
        for i_task in range(self.world.n_test):
            self.session.run(tf.global_variables_initializer())
            self.model.load(self.config.load, self.session)
            probs = np.zeros(self.world.n_test)
            probs[i_task] = 1
            updates = 0
            success = False
            while updates < 200:
                inst = [self.world.sample_test(probs) for _ in range(n_batch)]
                buf, rew, comp = _do_rollout(
                        self.config, self.world, inst, self.model, n_batch, self.session)
                self.objective.experience(buf)
                if not self.objective.ready():
                    continue
                updates += 1
                self.objective.train(self.session, repr_only=True)
                if np.mean(comp) == 1:
                    logging.info("[ad success] %d %d", i_task, updates)
                    success = True
                    break
                #if updates % 10 == 0:
                #    logging.info("[ad reward] %d %f", i_task, np.mean(rew))
                #    logging.info("[ad complete] %d %f", i_task, np.mean(comp))
            if not success:
                logging.info("[ad failure] %d %f %f", i_task, np.mean(rew), np.mean(comp))

        logging.info("")

#class RllAdaptationEvaluator(object):
#    def __init__(self, config, world, model, session):
#        self.config = config
#        self.session = session
#        env = TfEnv(RllEnvWrapper(world, use_val=True))
#        policy = RllPolicyWrapper(model, env.spec, env._wrapped_env, session)
#        baseline = LinearFeatureBaseline(env.spec)
#        algo_ctor = globals()[self.config.trainer.algo]
#        self.algo = algo_ctor(
#            env=env,
#            policy=policy,
#            baseline=baseline,
#            batch_size=self.config.objective.n_train_batch,
#            max_path_length=self.config.trainer.max_rollout_len,
#            discount=self.config.objective.discount,
#            step_size=self.config.objective.step_size,
#            entropy_bonus=self.config.objective.entropy_bonus,
#        )
#        self.model = model
#
#    def evaluate(self):
#        self.algo.start_worker()
#        self.session.run(tf.global_variables_initializer())
#        self.model.load(self.config.load, self.session)
#        for i_iter in range(20):
#            with rll_logger.prefix("EVAL itr #%d | " % i_iter):
#                paths = self.algo.obtain_samples(i_iter)
#                samples_data = self.algo.process_samples(i_iter, paths)
#                self.algo.log_diagnostics(paths)
#                self.algo.optimize_policy(i_iter, samples_data)
#                rll_logger.dump_tabular(with_prefix=True)
#        self.algo.shutdown_worker()
#        self.model.load(self.config.load, self.session)
