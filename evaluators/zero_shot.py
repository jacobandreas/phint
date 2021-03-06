from trainers.curriculum import _do_rollout

import logging
import numpy as np

class ZeroShotEvaluator(object):
    def __init__(self, config, world, model, session):
        self.config = config
        self.world = world
        self.model = model
        self.session = session

    def evaluate(self):
        logging.info("[ZERO SHOT EVAL]")
        self.model.load(self.config.load, self.session)

        n_batch = self.config.trainer.n_rollout_batch
        total_rew = 0.
        total_comp = 0.
        for i_task in range(self.world.n_test):
            probs = np.zeros(self.world.n_test)
            probs[i_task] = 1
            task_rew = 0.
            task_comp = 0.
            for i in range(5):
                inst = [self.world.sample_test(probs) for _ in range(n_batch)]
                buf, rew, comp = _do_rollout(
                        self.config, self.world, inst, self.model, n_batch, self.session)
                task_rew += np.mean(rew)
                task_comp += np.mean(comp)
            task_rew /= 5
            task_comp /= 5
            total_rew += task_rew
            total_comp += task_comp
            logging.info("[zs reward] %f", task_rew)
            logging.info("[zs complete] %f", task_comp)
        total_rew /= self.world.n_test
        total_comp /= self.world.n_test

        logging.info("[total zs reward] %f", total_rew)
        logging.info("[total zs comp] %f", total_comp)
        logging.info("")
