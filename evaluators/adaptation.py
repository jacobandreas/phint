from trainers.curriculum import _do_rollout

import logging
import numpy as np
import tensorflow as tf

class AdaptationEvaluator(object):
    def __init__(self, config, session):
        self.config = config
        self.session = session

    def evaluate(self, world, model, objective):
        logging.info("[ADAPTATION EVAL]")
        
        n_batch = self.config.trainer.n_rollout_batch
        for i_task in range(world.n_test):
            self.session.run(tf.global_variables_initializer())
            model.load(self.config.load, self.session)
            probs = np.zeros(world.n_test)
            probs[i_task] = 1
            updates = 0
            while updates < 200:
                inst = [world.sample_test(probs) for _ in range(n_batch)]
                buf, rew, comp = _do_rollout(
                        self.config, world, inst, model, n_batch, self.session)
                objective.experience(buf)
                if not objective.ready():
                    continue
                updates += 1
                objective.train(self.session)
                if updates % 10 == 0:
                    logging.info("[ad reward] %d %f", i_task, np.mean(rew))
                    logging.info("[ad complete] %d %f", i_task, np.mean(comp))

        logging.info("")
