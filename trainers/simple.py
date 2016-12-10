from misc.experience import Transition
from misc import util

from collections import defaultdict
import logging
import numpy as np
import time

N_ROLLOUT_BATCH = 100

class SimpleTrainer(object):
    def __init__(self, config):
        pass

    def train(self, world, model):
        i_iter = 0
        counts = defaultdict(lambda: 0.)
        rewards = defaultdict(lambda: 0.)

        while True:
            tasks = [world.sample_task() for _ in range(N_ROLLOUT_BATCH)]
            obs = world.reset(tasks)
            mstates = model.init([t.hint for t in tasks])
            stops = [False] * N_ROLLOUT_BATCH
            bufs = [[] for _ in range(N_ROLLOUT_BATCH)]
            total_rewards = np.zeros(N_ROLLOUT_BATCH)
            while max(len(b) for b in bufs) < 100 and not all(stops):
                actions, mstates_ = model.act(obs)
                obs_, rew, stops_ = world.step(actions, tasks)
                for i in range(N_ROLLOUT_BATCH):
                    if not stops[i]:
                        bufs[i].append(Transition(obs[i], mstates[i], actions[i],
                            obs_[i], mstates_[i], rew[i]))
                        total_rewards[i] += rew[i]
                        stops[i] = stops_[i]
                obs = obs_
                mstates = mstates_

            for buf in bufs:
                model.experience(buf)
            for task, rew in zip(tasks, total_rewards):
                counts[task.hint] += 1
                rewards[task.hint] += rew
                assert rew <= 1

            err = model.train()
            if err is not None:
                logging.info("[err] %d %f", i_iter, err)

            i_iter += 1

            if i_iter % 100 == 0:
                logging.info("[rewards %d]", N_ROLLOUT_BATCH * i_iter)
                total = 0.
                for hint in sorted(counts.keys()):
                    score = rewards[hint] / counts[hint]
                    logging.info("[reward %s] %f (%d)", util.pp_sexp(hint), 
                            score, counts[hint])
                    total += score
                logging.info("[reward mean] %f", total / len(counts))
                counts = defaultdict(lambda: 0.)
                rewards = defaultdict(lambda: 0.)

