from curriculum import _do_rollout
from misc.experience import Transition
from misc import util
import worlds

import logging
import numpy as np
import tensorflow as tf

class ImitationTrainer(object):
    def __init__(self, config, session):
        self.config = config
        self.session = session

    def train(self, world, model, objective):
        self.session.run(tf.global_variables_initializer())
        n_batch = self.config.trainer.n_rollout_batch

        err = 0
        i_iter = 0
        i_rollout = 0
        model.save(self.session)
        #while True:
        while i_iter < 10000:
            inst = [world.sample_train() for _ in range(n_batch)]
            buf = []
            mstate = model.init(inst, None)
            for it, m in zip(inst, mstate):
                demo = it.demo
                out = []
                for s, a, s_ in demo:
                    out.append(Transition(
                        s.features(), m, a, s_.features(), m, 0))
                buf.append(out)
            i_rollout += self.config.trainer.n_rollout_batch
            objective.experience(buf)
            if not objective.ready():
                continue
            err += objective.train(self.session)
            i_iter += 1

            n_update = self.config.trainer.n_update
            if i_iter % n_update == 0:
                logging.info("[iter] %d", i_iter)
                logging.info("[rollout] %d", i_rollout)
                logging.info("[step] %d", self.config.objective.n_train_batch * i_iter)
                logging.info("[err] %s", err / n_update)

                rew = 0
                for i in range(10):
                    inst = [world.sample_train() for _ in range(n_batch)]
                    rollouts, rew_here, _ = _do_rollout(self.config, world, inst, model, n_batch, self.session)
                    #for rollout, it in zip(rollouts[:1], inst[:1]):
                    #    print " ".join([str(r.a[0]) for r in rollout])
                    #    print " ".join([str(a) for s, a, s_ in it.demo])
                    #    print
                    rew += rew_here
                logging.info("[rew] %f", np.mean(rew) / 10)

                logging.info("")
                err = 0
                model.save(self.session)
