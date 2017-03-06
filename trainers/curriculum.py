from misc.experience import Transition
from misc import util

from collections import defaultdict
import logging
import numpy as np
import time
import tensorflow as tf


class CurriculumTrainer(object):
    def __init__(self, config):
        self.config = config
        self.session = tf.Session()

    def train(self, world, model, objective):
        self.session.run(tf.global_variables_initializer())
        n_batch = self.config.trainer.n_rollout_batch

        i_iter = 0
        counts = defaultdict(lambda: 0.)
        rewards = defaultdict(lambda: 0.)
        err = 0
        max_len = 1

        while True:
            try:
                task = [world.sample_task(max_len=max_len) for _ in range(n_batch)]
            except:
                max_len += 1
                continue

            buf, total_reward = self.do_rollout(world, task, model, n_batch)
            objective.experience(buf)
            for t, r in zip(task, total_reward):
                assert r <= 1
                counts[t.hint] += 1
                rewards[t.hint] += r

            if not objective.ready():
                continue
            err += objective.train(self.session)
            i_iter += 1

            examples = [[t.a for t in b] for b in buf[:3]]

            n_update = self.config.trainer.n_update
            if i_iter % n_update == 0:
                logging.info("[err] %d %s", i_iter, err / n_update)
                logging.info("[rewards %d]", self.config.objective.n_train_batch * i_iter)
                min_score = 1
                for hint in sorted(counts.keys()):
                    score = rewards[hint] / counts[hint]
                    logging.info("[reward %s] %f (%d)", util.pp_sexp(hint), 
                            score, counts[hint])
                    min_score = min(score, min_score)
                logging.info("\n"+"\n".join([str(e) for e in examples]))
                if min_score > 0.8:
                    max_len += 1
                counts = defaultdict(lambda: 0.)
                rewards = defaultdict(lambda: 0.)
                err = 0

    def do_rollout(self, world, task, model, n_batch):
        obs = world.reset(task)
        mstate = model.init(task, obs)
        stop = [False] * n_batch
        buf = [[] for _ in range(n_batch)]
        total_reward = np.zeros(n_batch)
        while max(len(b) for b in buf) < self.config.trainer.max_rollout_len and not all(stop):
            action, agent_stop, mstate_ = model.act(obs, mstate, task, self.session)
            world_action, _  = zip(*action)
            obs_, rew, world_stop = world.step(world_action, task)
            complete_rew = world.complete(task)
            for i in range(n_batch):
                if not stop[i]:
                    rew_here = rew[i]
                    if agent_stop[i]:
                        rew_here += complete_rew[i]
                    total_reward[i] += rew_here
                    buf[i].append(Transition(obs[i], mstate[i], action[i],
                        obs_[i], mstate_[i], rew_here))
                    stop[i] = stop[i] or agent_stop[i] or world_stop[i]
            obs = obs_
            mstate = mstate_
        return buf, total_reward
