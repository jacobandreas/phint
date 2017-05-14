from misc import array, util

from collections import namedtuple
import numpy as np

N_KINDS = 6
N_SPLIT = 1

DummyTask = namedtuple("DummyTask", ["id", "goal", "hint"])

class DummyInstance(object):
    def __init__(self, task, init_state):
        self.task = task
        self.state = init_state

class DummyWorld(object):
    def __init__(self, config):
        self.config = config
        self.tasks = []
        self.vocab = util.Index()
        self.random = util.next_random()

        for k in range(N_KINDS):
            for s in range(N_SPLIT):
                self.vocab.index("%d_%d" % (k, s))

        i = 0
        for k1 in range(N_KINDS):
            for k2 in range(k1+1, N_KINDS):
                self.tasks.append(DummyTask(i, frozenset([k1, k2]), None))
                i += 1

        ids = list(range(len(self.tasks)))
        self.random.shuffle(ids)
        self.train_ids = ids[:len(ids)/2]
        #self.test_ids = ids[len(self.train_ids):]
        self.test_ids = ids

        self.n_act = N_KINDS
        self.n_vocab = len(self.vocab)
        self.n_obs = N_KINDS * N_SPLIT

        self.n_tasks = len(self.tasks)
        self.n_train = len(self.train_ids)
        self.n_test = len(self.test_ids)
        self.is_discrete = True
        self.max_hint_len = 2

    def sample_train(self, p=None):
        if p is not None:
            assert len(p) == self.n_train
        else:
            p = np.ones(self.n_train) / self.n_train
        return self.sample_instance(self.random.choice(self.train_ids, p=p))

    def sample_test(self, p=None):
        if p is not None:
            assert len(p) == self.n_test
        else:
            p = np.ones(self.n_test) / self.n_test
        return self.sample_instance(self.random.choice(self.test_ids, p=p))

    def sample_instance(self, task_id):
        task = self.tasks[task_id]
        split = self.random.randint(N_SPLIT)
        indexed_steps = [
                self.vocab["%d_%d" % (w, split)]
                for w in task.goal]
        task = task._replace(hint=tuple(indexed_steps))
        state = self.sample_state(task, split)
        return DummyInstance(task, state)

    def sample_state(self, task, split):
        return DummyState(np.zeros(self.n_act), task.goal, split)

    def reset(self, insts):
        return [inst.state.features() for inst in insts]

    def step(self, actions, insts):
        features, rewards, stops = [], [], []
        for action, inst in zip(actions, insts):
            r, s, stop = inst.state.step(action)
            inst.state = s
            features.append(s.features())
            rewards.append(r)
            stops.append(stop)
        return features, rewards, stops

class DummyState(object):
    def __init__(self, used, goal, pos):
        self.used = used
        self.goal = goal
        self.pos = pos

        self.feats = np.zeros(used.size * N_SPLIT)
        self.feats[used.size*pos:used.size*(pos+1)] = used

    def features(self):
        return self.feats

    def step(self, action):
        new_used = self.used.copy()
        if new_used[action] == 0:
            new_used[action] += 1
        if all(new_used[g] for g in self.goal):
            reward = 1
            stop = True
        elif new_used.sum() >= 2:
            reward = 0
            stop = True
        else:
            reward = 0
            stop = False
        return reward, DummyState(new_used, self.goal, self.pos), stop
