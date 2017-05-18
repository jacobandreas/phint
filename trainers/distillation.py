from curriculum import _do_rollout
from misc.experience import Transition
from misc import util
import worlds

from collections import defaultdict
import logging
import numpy as np
import os
import pickle
import time
import tensorflow as tf

class DistillationTrainer(object):
    def __init__(self, config, session):
        self.config = config
        self.session = session
        self.random = util.next_random()

    def train(self, world, model, objective, eval_thunk=None):
        n_batch = self.config.trainer.n_rollout_batch
        n_update = self.config.trainer.n_update

        self.session.run(tf.global_variables_initializer())
        model.save(self.session)

        for i_train in range(world.n_train):
            logging.info("task %d" % i_train)
            probs = np.zeros(world.n_train)
            probs[i_train] = 1
            task = world.sample_train(probs).task
            logging.info(str(task))
            model.load(self.config.name, self.session)

            successes = []
            for i in range(10):
                total_reward = 0
                for j in range(n_update):
                    inst = [world.sample_train(probs) for _ in range(n_batch)]
                    bufs, rewards, _ = _do_rollout(self.config, world, inst,
                            model, n_batch, self.session)
                    total_reward += rewards.sum()
                    if rewards.sum() > .9 * n_batch:
                        for it, r, buf in zip(inst, rewards, bufs):
                            # TODO criterion
                            if r > 0:
                                successes.append((it.task, buf))
                    objective.experience(bufs)
                    if not objective.ready():
                        continue
                    objective.train(self.session)
                logging.info("%d %f" % (i, total_reward / n_update))
            logging.info(str(len(successes)))

            with open(
                    os.path.join(
                        self.config.experiment_dir,
                        "paths.%d.pkl" % i_train), 
                    "wb") as pickle_f:
                pickle.dump(successes, pickle_f, pickle.HIGHEST_PROTOCOL)
