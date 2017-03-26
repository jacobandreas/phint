from misc import util

from collections import namedtuple
import numpy as np
import sexpdata
import os
from nltk.tokenize import TreebankWordTokenizer
import logging

EXAMPLE_DIR = "worlds/shurdlurn/examples"

def listify(sexp):
    out = []
    assert sexp[0] == sexpdata.Symbol("string")
    brackets = sexp[1].value()
    if brackets == "undefined" or brackets == "badjava:":
        raise Exception("unable to listify")
    for bracket in brackets:
        if isinstance(bracket, sexpdata.Symbol):
            continue
        if isinstance(bracket, str):
            print "BRACKET IS STR!"
            print brackets
            exit()
        if len(bracket.value()) == 0:
            out.append(())
            continue
        inner = bracket.value()[0]
        if isinstance(inner, sexpdata.Symbol):
            out.append(tuple(int(i) for i in inner.value().split(",")))
            continue
        assert isinstance(inner, int)
        out.append((inner,))
    return tuple(out)

ShurdlurnTask = namedtuple("ShurdlurnTask", ["start", "end", "hint", "id"])
#ShurdlurnInstance = namedtuple("ShurdlurnInstance", ["task", "state"])
class ShurdlurnInstance(object):
    def __init__(self, task, state):
        self.task = task
        self.state = state

class ShurdlurnWorld(object):
    def __init__(self, config):
        self.config = config

        with open(os.path.join(
                EXAMPLE_DIR, config.world.train_fold + ".ids.txt")) as tr_id_f:
            train_names = {s.strip() for s in tr_id_f}
        with open(os.path.join(
                EXAMPLE_DIR, config.world.test_fold + ".ids.txt")) as te_id_f:
            test_names = {s.strip() for s in te_id_f}
        self.train_ids = []
        self.test_ids = []

        example_names = os.listdir(EXAMPLE_DIR)
        tasks = {}
        self.vocab = util.Index()
        self.task_index = util.Index()
        tokenizer = TreebankWordTokenizer()
        for file_name in example_names:
            if not file_name.endswith(".lisp"):
                continue
            name = file_name[:-5]
            if not (name in train_names or name in test_names):
                continue
            with open(os.path.join(EXAMPLE_DIR, file_name)) as example_f:
                try:
                    examples = sexpdata.parse(example_f.read())
                except Exception as e:
                    logging.warn("unable to parse %s", name)
                for i_example, example in enumerate(examples):
                    try:
                        start_outer = example[2][2]
                        end_outer = example[7]
                        utt_outer = example[5]
                        assert start_outer[0] == sexpdata.Symbol("graph")
                        assert end_outer[0] == sexpdata.Symbol("targetValue")
                        assert utt_outer[0] == sexpdata.Symbol("utterance")
                        start = listify(start_outer[2][0])
                        end = listify(end_outer[1])
                        utt = utt_outer[1]
                        if isinstance(utt, int):
                            utt = str(utt)
                        elif isinstance(utt, sexpdata.Symbol):
                            utt = utt.value()
                        else:
                            assert isinstance(utt, str), utt
                    except Exception as e:
                        logging.warn("unable to process utt from %s", name)
                        continue
                    if start == end:
                        continue
                    utt = ["<s>"] + tokenizer.tokenize(utt.lower()) + ["</s>"]
                    #utt = [name + str(i_example)]
                    utt = tuple(self.vocab.index(tok) for tok in utt)
                    t_id = self.task_index.index((name, i_example))
                    if name in train_names:
                        self.train_ids.append(t_id)
                    elif name in test_names:
                        self.test_ids.append(t_id)
                    else:
                        assert False
                    task = ShurdlurnTask(start, end, utt, t_id)
                    tasks[t_id] = task

        #self.train_ids = self.train_ids[:100]
        #self.test_ids = self.test_ids[:100]

        logging.info("loaded %d train utts from %d sessions", len(self.train_ids), len(train_names))
        logging.info("loaded %d test utts from %d sessions", len(self.test_ids), len(test_names))

        max_width = max(len(t.start) for t in tasks.values())
        max_height = max(
                max(max(len(s) for s in t.start) for t in tasks.values()),
                max(max(len(s) for s in t.end) for t in tasks.values()))
        n_kinds = 1 + max(
                max(max(max(s) if len(s) > 0 else 0 for s in t.start) for t in tasks.values()),
                max(max(max(s) if len(s) > 0 else 0 for s in t.end) for t in tasks.values()))

        self.max_width = max_width
        self.max_height = max_height
        self.max_hint_len = max(len(t.hint) for t in tasks.values())
        self.n_kinds = n_kinds
        self.n_obs = max_width * max_height * (n_kinds + 1) + max_width
        self.n_act = 2 + 1 + n_kinds
        self.n_tasks = len(tasks)
        self.n_train = len(self.train_ids)
        self.n_test = len(self.test_ids)
        self.tasks = tasks
        self.random = util.next_random()

    def sample_train(self, p=None):
        return self.sample_instance(self.train_ids, p)

    def sample_test(self, p=None):
        return self.sample_instance(self.test_ids, p)

    def sample_instance(self, fold, p):
        assert p is None or len(p) == len(fold)
        idx_task = self.random.choice(len(fold), p=p)
        id_task = fold[idx_task]
        task = self.tasks[id_task]
        init_state = ShurdlurnState(0, task.start, task.end, self.max_width, self.max_height, self.n_kinds)
        return ShurdlurnInstance(task, init_state)

    def reset(self, tasks):
        return [t.state.features() for t in tasks]

    def step(self, actions, insts):
        features = []
        rewards = []
        stops = []
        for a, t in zip(actions, insts):
            reward, nstate, stop = t.state.step(a)
            t.state = nstate
            features.append(nstate.features())
            rewards.append(reward)
            stops.append(stop)
        return features, rewards, stops

    def complete(self, insts):
        return [0] * len(insts)
        #return [1 if i.state.blocks == i.state.goal else 0 for i in insts]

class ShurdlurnState(object):
    def __init__(self, agent_x, blocks, goal, max_width, max_height, n_kinds):
        self.agent_x = agent_x
        self.blocks = blocks
        self.goal = goal
        self.max_width = max_width
        self.max_height = max_height
        self.n_kinds = n_kinds
        self._cached_features = None

    def features(self):
        if self._cached_features is None:
            board = np.zeros((self.max_width, self.max_height, self.n_kinds+1))
            pos = np.zeros((self.max_width))
            pos[self.agent_x] = 1
            for x in range(len(self.blocks)):
                for y in range(len(self.blocks[x])):
                    assert y >= 0
                    board[x, y, self.blocks[x][y]+1] = 1
            self._cached_features = np.concatenate((board.ravel(), pos))
        return self._cached_features

    def step(self, action):
        if action == 0:
            nx = self.agent_x - 1
        elif action == 1:
            nx = self.agent_x + 1
        else:
            nx = self.agent_x
        nx = max(nx, 0)
        nx = min(nx, self.max_width - 1)

        new_blocks = []
        reward = 0
        for x in range(len(self.blocks)):
            if x == self.agent_x and action == 2:
                new_blocks.append(self.blocks[x][:-1])
                if len(self.goal[x]) < len(self.blocks[x]):
                    reward += 1
                else:
                    reward -= 1
            elif (x == self.agent_x 
                    and action > 2 
                    and action < self.n_kinds + 2 + 1
                    and len(self.blocks[x]) < self.max_height - 1):
                kind = action - 2 - 1
                assert kind < self.n_kinds
                new_blocks.append(self.blocks[x] + (kind,))
                if len(self.goal[x]) > len(self.blocks[x]) and kind == self.goal[x][len(self.blocks[x])]:
                    reward += 1
                else:
                    reward -= 1
            else:
                new_blocks.append(self.blocks[x])
        new_blocks = tuple(new_blocks)
        nstate = ShurdlurnState(nx, new_blocks, self.goal, self.max_width, self.max_height, self.n_kinds)
        stop = new_blocks == self.goal
        if stop:
            reward += 10
        return reward / 10., nstate, stop
