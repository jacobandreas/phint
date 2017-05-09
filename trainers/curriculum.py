from misc.experience import Transition
from misc import util
import worlds

from collections import defaultdict
import logging
import numpy as np
import time
import tensorflow as tf

def _do_rollout(config, world, task, model, n_batch, session, intrinsic=False):
    obs = world.reset(task)
    mstate = model.init(task, obs)
    stop = [False] * n_batch
    buf = [[] for _ in range(n_batch)]
    total_reward = np.zeros(n_batch)
    total_complete = np.zeros(n_batch)
    while max(len(b) for b in buf) < config.trainer.max_rollout_len and not all(stop):
        action, agent_stop, mstate_, intrinsic_rew = model.act(
                obs, mstate, task, session)
        world_action, _  = zip(*action)
        obs_, world_rew, world_stop = world.step(world_action, task)
        rew = np.asarray(world_rew) 
        if intrinsic:
            rew += config.model.intrinsic_bonus * np.asarray(intrinsic_rew)
        #complete_rew = world.complete(task)
        for i in range(n_batch):
            if not stop[i]:
                rew_here = rew[i]
                #total_complete[i] += complete_rew[i]
                #if agent_stop[i]:
                #    rew_here += complete_rew[i]
                total_reward[i] += rew_here
                buf[i].append(Transition(obs[i], mstate[i], action[i],
                    obs_[i], mstate_[i], rew_here))
                stop[i] = stop[i] or agent_stop[i] or world_stop[i]
        obs = obs_
        mstate = mstate_
    return buf, total_reward, total_complete

class CurriculumTrainer(object):
    def __init__(self, config, session):
        self.config = config
        self.session = session
        self.random = util.next_random()

    #def _recompute_task_probs(self, world, counts, rewards, max_len):
    #    probs = np.zeros(world.n_tasks)
    #    max_reward = max(rewards[t] / (1 + counts[t]) for t in world.tasks)
    #    for i, task in enumerate(world.tasks):
    #        if len(task.hint) > max_len:
    #            continue
    #        probs[i] = 1 - rewards[task] / counts[task]
    #        #probs[i] = (0.05 + max_reward) / (0.05 + rewards[task] / (1 + counts[task]))
    #    if not probs.any():
    #        return None
    #    probs /= np.sum(probs)
    #    return probs

    def _recompute_task_probs(self, world, counts, rewards, max_len):
        probs = np.zeros(world.n_train)
        indices = list(range(world.n_train))
        self.random.shuffle(indices)
        indices = indices[:100]
        for i in indices:
            probs[i] = 1
        probs /= np.sum(probs)
        return probs

    def train(self, world, model, objective, eval_thunk=None):
        self.session.run(tf.global_variables_initializer())
        n_batch = self.config.trainer.n_rollout_batch

        i_iter = 0
        i_rollout = 0
        counts = defaultdict(lambda: 1.) # ick
        rewards = defaultdict(lambda: 0.)
        err = 0
        #max_len = min(len(task.hint) for task in world.tasks)
        max_len = 100
        task_probs = self._recompute_task_probs(world, counts, rewards, max_len)
        assert task_probs is not None
        #task_probs = None

        model.save(self.session)
        while True:
            inst = [world.sample_train(task_probs) for _ in range(n_batch)]
            #inits = [str(i.state.blocks) + " " + str(i.state.goal) for i in inst]
            buf, total_reward, _ = _do_rollout(self.config, world, inst, model, n_batch, self.session, intrinsic=True)
            i_rollout += self.config.trainer.n_rollout_batch
            objective.experience(buf)
            for i, it, r in zip(range(len(inst)), inst, total_reward):
                #assert r <= 1
                counts[it.task] += 1
                rewards[it.task] += r

            if not objective.ready():
                continue
            err += objective.train(self.session)
            i_iter += 1

            examples = [[t.a for t in b] for b in buf[:3]]
            examples = [" ".join(
                    [str(a) for a, r in e]) 
                    #[str(a) if r == 0 else str(a) + " _" for a, r in e]) 
                for e in examples]

            n_update = self.config.trainer.n_update
            if i_iter % n_update == 0:
                logging.info("[iter] %d", i_iter)
                logging.info("[rollout] %d", i_rollout)
                logging.info("[step] %d", self.config.objective.n_train_batch * i_iter)
                logging.info("[err] %s", err / n_update)
                scores = []
                for hint in sorted(counts.keys()):
                    score = rewards[hint] / counts[hint]
                    scores.append(score)
                    if self.config.trainer.log_each_task:
                        logging.info(
                                "[reward %s] %f (%d)",
                                util.pp_sexp(hint), score, counts[hint])
                for i_ex, ex in enumerate(examples):
                    logging.info("[rollout %d] %s" % (i_ex, ex))
                    #logging.info("[goal %d] %s" % (i_ex, inits[i_ex]))
                if min(scores) > 0.8:
                    max_len += 1
                model.save(self.session)

                if (i_iter % (n_update * self.config.trainer.resample_every) == 0
                        or i_iter > self.config.trainer.switch_iter):
                    task_probs = self._recompute_task_probs(world, counts, rewards, max_len)

                #logging.info("[probs] %s", task_probs)
                logging.info("[mean] %f" % np.mean(scores))
                logging.info("")
                counts = defaultdict(lambda: 1.)
                rewards = defaultdict(lambda: 0.)
                err = 0

                if i_iter % (n_update * 10) == 0 and eval_thunk is not None:
                    eval_thunk()
                    model.load(self.config.name, self.session)
