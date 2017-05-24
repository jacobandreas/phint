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
    def __init__(self, config, world, model, objective, describer, session):
        self.config = config
        self.world = world
        self.model = model
        self.objective = objective
        self.describer = describer
        self.session = session

    def evaluate(self):
        logging.info("[ADAPTATION EVAL]")
        
        n_batch = self.config.trainer.n_rollout_batch
        for i_task in range(self.world.n_test):
            self.session.run(tf.global_variables_initializer())
            self.model.load(self.config.load, self.session)
            probs = np.zeros(self.world.n_test)
            probs[i_task] = 1

            logging.info("[hint guessing]")
            hint_results = {}
            for i_guess in range(200):
                hint, = self.describer.sample(1)
                while hint in hint_results:
                    hint, = self.describer.sample(1)
                inst = [self.world.sample_test(probs) for _ in range(20)]
                for it in inst:
                    it.task = it.task._replace(hint=hint)

                buf, rew, comp = _do_rollout(
                        self.config, self.world, inst, self.model, 20,
                        self.session)
                hint_results[hint] = np.mean(rew)

            best_hint = max(hint_results, key=lambda h: hint_results[h])
            gold_hint = self.world.sample_test(probs).task.hint
            logging.info("[guess] %s %f", 
                    " ".join([self.world.vocab.get(w) for w in best_hint]),
                    hint_results[best_hint])
            logging.info("[gold] %s",
                    " ".join([self.world.vocab.get(w) for w in gold_hint]))

            #self.session.run([self.model.o_reset_temp])
            updates = 0
            success = False
            while updates < 300:
                total_rew = 0
                for i in range(5):
                    inst = [self.world.sample_test(probs) for _ in range(n_batch)]
                    for it in inst:
                        it.task = it.task._replace(hint=best_hint)
                    buf, rew, comp = _do_rollout(
                            self.config, self.world, inst, self.model, n_batch, self.session)
                    total_rew += np.mean(rew)
                    self.objective.experience(buf)
                    if not self.objective.ready():
                        continue
                    updates += 1
                    self.objective.train(self.session, repr_only=True)

                total_rew /= 5
                logging.info("[%02d] %f" % (updates, total_rew))
                if np.mean(comp) == 1:
                    logging.info("[ad success] %d %d", i_task, updates)
                    success = True
                    break
                #if updates % 10 == 0:
                #    logging.info("[ad reward] %d %f", i_task, np.mean(rew))
                #    logging.info("[ad complete] %d %f", i_task, np.mean(comp))
            if not success:
                logging.info("[ad failure] %d %f %f", i_task, total_rew, np.mean(comp))

        logging.info("")
        self.model.load(self.config.load, self.session)

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
