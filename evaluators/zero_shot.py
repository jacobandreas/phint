from trainers.curriculum import _do_rollout

import logging
import numpy as np

class ZeroShotEvaluator(object):
    def __init__(self, config, session):
        self.config = config
        self.session = session

    def evaluate(self, world, model):
        logging.info("[ZERO SHOT EVAL]")
        model.load(self.config.load, self.session)

        n_batch = self.config.trainer.n_rollout_batch
        total_rew = 0.
        for i_task in range(world.n_test):
            probs = np.zeros(world.n_test)
            probs[i_task] = 1
            inst = [world.sample_test(probs) for _ in range(n_batch)]
            buf, rew = _do_rollout(
                    self.config, world, inst, model, n_batch, self.session)
            total_rew += np.mean(rew)
            logging.info("[zs reward] %f", np.mean(rew))
        total_rew /= world.n_test

        logging.info("[total zs reward] %f", total_rew)
        logging.info("")