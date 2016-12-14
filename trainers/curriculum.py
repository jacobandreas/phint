from misc.experience import Transition
from misc import util

from collections import defaultdict
import logging
import numpy as np
import time

N_ROLLOUT_BATCH = 50

class CurriculumTrainer(object):
    def __init__(self, config):
        pass

    def train(self, world, model):
        i_iter = 0
        counts = defaultdict(lambda: 0.)
        rewards = defaultdict(lambda: 0.)
        err = 0
        max_len = 1

        while True:
            try:
                tasks = [world.sample_task(max_len=max_len) for _ in range(N_ROLLOUT_BATCH)]
            except:
                max_len += 1
                continue
            obs = world.reset(tasks)
            mstates = model.init([t.hint for t in tasks])
            stops = [False] * N_ROLLOUT_BATCH
            bufs = [[] for _ in range(N_ROLLOUT_BATCH)]
            total_rewards = np.zeros(N_ROLLOUT_BATCH)
            while max(len(b) for b in bufs) < 100 and not all(stops):
                actions, probs, mstates_, agent_stops = model.act(obs)
                obs_, rew, world_stops = world.step(actions, tasks)
                complete_rew = world.complete(tasks)
                for i in range(N_ROLLOUT_BATCH):
                    if not stops[i]:
                        rew_here = rew[i]
                        if agent_stops[i]:
                            rew_here += complete_rew[i]
                        total_rewards[i] += rew_here
                        bufs[i].append(Transition(obs[i], mstates[i], actions[i],
                            obs_[i], mstates_[i], rew_here, probs[i]))
                        stops[i] = agent_stops[i] or world_stops[i]
                        assert mstates[i].index < len(mstates[i].hint)

                #print np.asarray(stops).astype(int)
                #print

                obs = obs_
                mstates = mstates_

            for buf in bufs:
                model.experience(buf)
            for task, rew in zip(tasks, total_rewards):
                assert rew <= 1
                counts[task.hint] += 1
                rewards[task.hint] += rew

            e = model.train()
            if e is not None:
                err += e
            else:
                continue

            i_iter += 1

            if i_iter % 10 == 0:
                logging.info("[err] %d %s", i_iter, err)
                logging.info("[rewards %d]", N_ROLLOUT_BATCH * i_iter)
                min_score = 1
                for hint in sorted(counts.keys()):
                    score = rewards[hint] / counts[hint]
                    logging.info("[reward %s] %f (%d)", util.pp_sexp(hint), 
                            score, counts[hint])
                    min_score = min(score, min_score)
                if min_score > 0.8:
                    max_len += 1
                counts = defaultdict(lambda: 0.)
                rewards = defaultdict(lambda: 0.)
                err = 0

