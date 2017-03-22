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

ShurdlurnTask = namedtuple("ShurdlurnTask", ["start", "end", "hint"])
ShurdlurnInstance = namedtuple("ShurdlurnInstance", ["task", "state"])

class ShurdlurnWorld(object):
    def __init__(self, config):
        self.config = config
        example_names = os.listdir(EXAMPLE_DIR)[:10]
        tasks = []
        self.vocab = util.Index()
        tokenizer = TreebankWordTokenizer()
        for name in example_names:
            if not name.endswith(".lisp"):
                continue
            with open(os.path.join(EXAMPLE_DIR, name)) as example_f:
                try:
                    examples = sexpdata.parse(example_f.read())
                except Exception as e:
                    logging.warn("unable to parse %s", name)
                for example in examples:
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
                    utt = ["<s>"] + tokenizer.tokenize(utt.lower()) + ["</s>"]
                    utt = tuple(self.vocab.index(tok) for tok in utt)
                    task = ShurdlurnTask(start, end, utt)
                    tasks.append(task)

        max_width = max(len(t.start) for t in tasks)
        max_height = max(
                max(max(len(s) for s in t.start) for t in tasks),
                max(max(len(s) for s in t.end) for t in tasks))
        n_kinds = 1 + 1 + max(
                max(max(max(s) if len(s) > 0 else 0 for s in t.start) for t in tasks),
                max(max(max(s) if len(s) > 0 else 0 for s in t.end) for t in tasks))

        self.max_width = max_width
        self.max_height = max_height
        self.max_hint_len = max(len(t.hint) for t in tasks)
        self.n_kinds = n_kinds
        self.n_obs = max_width * max_height * n_kinds + max_width
        self.n_act = 2 + 1 + (n_kinds - 1)
        self.n_tasks = len(tasks)
        self.tasks = tasks
        self.random = util.next_random()

    def sample_instance(self, p=None):
        assert p is None or len(p) == len(self.tasks)
        i_task = self.random.choice(len(self.tasks), p=p)
        task = self.tasks[i_task]
        init_state = ShurdlurnState(0, task.start, self.max_width, self.max_height, self.n_kinds)
        return ShurdlurnInstance(task, init_state)

    def reset(self, tasks):
        return [t.state.features() for t in tasks]

    def step(self, actions, insts):
        features = []
        rewards = []
        stops = []
        for a, t in zip(actions, insts):
            reward, nstate = t.state.step(a)
            t.state = nstate
            stop = False
            features.append(nstate.features())
            rewards.append(reward)
            stops.append(stop)
        return features, rewards, stops

class ShurdlurnState(object):
    def __init__(self, agent_x, blocks, max_width, max_height, n_kinds):
        self.agent_x = agent_x
        self.blocks = blocks
        self.max_width = max_width
        self.max_height = max_height
        self.n_kinds = n_kinds
        self._cached_features = None

    def features(self):
        if self._cached_features is None:
            board = np.zeros((self.max_width, self.max_height, self.n_kinds))
            pos = np.zeros((self.max_width))
            pos[self.agent_x] = 1
            for x in range(len(self.blocks)):
                for y in range(len(self.blocks[x])):
                    board[x, y, self.blocks[x][y]+1] = 1
            self._cached_features = np.concatenate((board.ravel(), pos))
        return self._cached_features

    def step(self, action):
        if action == 0:
            nx = self.agent_x - 1
        elif action == 1:
            nx = self.agent_x + 1
        self.agent_x = max(self.agent_x, 0)
        self.agent_x = min(self.agent_x, self.max_width - 1)

        new_blocks = []
        for x in range(len(self.blocks)):
            if x == self.agent_x and action == 2:
                new_blocks.append(self.blocks[x][:-1])
            elif x == self.agent_x and action > 2:
                kind = action - 2
                assert kind < self.n_kinds
                new_blocks.append(self.blocks[x] + (kind,))
            else:
                new_blocks.append(self.blocks[x])
        new_blocks = tuple(new_blocks)
        return 0, ShurdlurnState(nx, new_blocks, self.max_width, self.max_height, self.n_kinds)
