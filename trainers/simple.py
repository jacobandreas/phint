from misc.experience import Transition
from misc import util

from collections import defaultdict
import logging
import numpy as np
import time

class SimpleTrainer(object):
    def __init__(self, config):
        pass

    def train(self, world, model):
        i_iter = 0
        counts = defaultdict(lambda: 0.)
        rewards = defaultdict(lambda: 0.)

        while True:
            task = world.sample_task()
            print task.hint
            obs = world.reset([task])
            mstate = model.init([task.hint])[0]
            stop = False
            total_reward = 0
            while not stop:
                buf = []
                while len(buf) < 1000 and not stop:
                    action, mstate_ = model.act([obs])[0]
                    obs_, rew, stop = world.step([action], [task])
                    buf.append(Transition(obs, mstate, action, obs_, mstate_, rew))
                    obs = obs_
                    mstate = mstate_
                    total_reward += rew
                model.experience(buf)
                err = model.train()
                stop = True # TODO can we just pause the mission timer?
                if err is not None:
                    logging.info("[err] %f", err)

            counts[task.hint] += 1
            rewards[task.hint] += total_reward
            if (i_iter + 1) % 100 == 0:
                logging.info("[rewards]")
                for hint in sorted(counts.keys()):
                    logging.info("[reward %s] %f (%d)", util.pp_sexp(hint), 
                            rewards[hint] / counts[hint], counts[hint])
                counts = defaultdict(lambda: 0.)
                rewards = defaultdict(lambda: 0.)

            i_iter += 1
