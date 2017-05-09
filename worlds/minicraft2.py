from misc import array, util

from collections import namedtuple
import numpy as np
from skimage.measure import block_reduce

SIZE = 11
WINDOW_SIZE = 5

INGREDIENTS = ["wood", "ore", "grass", "stone"]

CRAFTS = [
    "stick", "metal", "rope", "shovel", "ladder", "axe", "trap", "sword",
    "bridge"
]

RECIPES = {
    "stick": {"wood"},
    "metal": {"ore"},
    "rope": {"grass"},
    "shovel": {"stick", "metal"},
    "ladder": {"stick", "rope"},
    "axe": {"stick", "stone"},
    "trap": {"metal", "rope"},
    "sword": {"metal", "stone"},
    "bridge": {"rope", "stone"}
}

HINTS = [
    ("wood", ["wood"]),
    ("ore", ["ore"]),
    ("grass", ["grass"]),
    ("stone", ["stone"]),
    ("stick", ["wood", "craft"]),
    ("metal", ["ore", "craft"]),
    ("rope", ["grass", "craft"]),
    ("shovel", ["stick", "metal", "craft"]),
    ("ladder", ["stick", "rope", "craft"]),
    ("axe", ["stick", "stone", "craft"]),
    ("trap", ["metal", "rope", "craft"]),
    ("sword", ["metal", "stone", "craft"]),
    ("bridge", ["rope", "stone", "craft"]),
]

TEST_IDS = list(range(len(HINTS))[::3])
TRAIN_IDS = [i for i in range(len(HINTS)) if i not in TEST_IDS]

N_ACTIONS = 6
UP, DOWN, LEFT, RIGHT, USE, CRAFT = range(N_ACTIONS)

Minicraft2Task = namedtuple("Minicraft2Task", ["id", "goal", "hint"])

def random_free(grid, random):
    pos = None
    while pos is None:
        (x, y) = (random.randint(SIZE), random.randint(SIZE))
        if grid[x, y, :].any():
            continue
        pos = (x, y)
    return pos

def neighbors(pos, dir=None):
    x, y = pos
    neighbors = []
    if x > 0 and (dir is None or dir == LEFT):
        neighbors.append((x-1, y))
    if y > 0 and (dir is None or dir == DOWN):
        neighbors.append((x, y-1))
    if x < SIZE - 1 and (dir is None or dir == RIGHT):
        neighbors.append((x+1, y))
    if y < SIZE - 1 and (dir is None or dir == UP):
        neighbors.append((x, y+1))
    return neighbors

class Minicraft2Instance(object):
    def __init__(self, task, init_state):
        self.task = task
        self.state = init_state


class Minicraft2World(object):
    def __init__(self, config):
        self.config = config
        self.tasks = []
        self.index = util.Index()
        self.vocab = util.Index()
        self.ingredients = [self.index.index(k) for k in INGREDIENTS]
        self.crafts = [self.index.index(k) for k in CRAFTS]
        self.recipes = {
            self.index.index(k): set(self.index.index(vv) for vv in v)
            for k, v in RECIPES.items()
        }
        self.hints = [
            (self.index.index(k), tuple(self.vocab.index(vv) for vv in v))
            for k, v in HINTS
        ]

        self.kind_to_obs = {}
        for k in self.ingredients:
            self.kind_to_obs[k] = len(self.kind_to_obs)

        self.n_obs = (
            2 * WINDOW_SIZE * WINDOW_SIZE * len(self.kind_to_obs)
            + len(self.index)
            + 4)
        self.n_act = N_ACTIONS
        self.is_discrete = True

        self.max_hint_len = 3
        self.n_vocab = len(self.vocab)
        self.random = util.next_random()

        self.tasks = []
        for i, (goal, steps) in enumerate(self.hints):
            self.tasks.append(Minicraft2Task(i, goal, steps))
        self.n_tasks = len(self.tasks)
        self.n_train = len(TRAIN_IDS)
        self.n_val = 0
        self.n_test = len(TEST_IDS)

    def sample_instance(self, task_id):
        task = self.tasks[task_id]
        state = self.sample_state(task)
        return Minicraft2Instance(task, state)

    def sample_state(self, task):
        grid = np.zeros((SIZE, SIZE, len(self.kind_to_obs)))
        for k in self.ingredients:
            obs = self.kind_to_obs[k]
            for _ in range(2):
                x, y = random_free(grid, self.random)
                grid[x, y, obs] = 1

        init_pos = random_free(grid, self.random)
        init_dir = self.random.randint(4)
        return Minicraft2State(self, grid, init_pos, init_dir, np.zeros(len(self.index)), task)

    def sample_train(self, p=None):
        return self.sample_instance(self.random.choice(TRAIN_IDS))

    def sample_val(self, p=None):
        assert False
        return self.sample_instance(self.random.choice(TRAIN_IDS))

    def sample_test(self, p=None):
        return self.sample_instance(self.random.choice(TEST_IDS))

    def reset(self, insts):
        return [inst.state.features() for inst in insts]

    def step(self, actions, insts):
        features, rewards, stops = [], [], []
        for action, inst in zip(actions, insts):
            reward, new_state, stop = inst.state.step(action)
            inst.state = new_state
            features.append(new_state.features())
            rewards.append(reward)
            stops.append(stop)
        return features, rewards, stops

class Minicraft2State(object):
    def __init__(self, world, grid, pos, dir, inventory, task):
        self.world = world
        self.grid = grid
        self.pos = pos
        self.dir = dir
        self.inventory = inventory
        self.task = task
        self._cached_features = None

    def features(self):
        if self._cached_features is not None:
            return self._cached_features

        x, y = self.pos
        hs = WINDOW_SIZE / 2
        bhs = WINDOW_SIZE * WINDOW_SIZE / 2

        grid_feats = array.pad_slice(self.grid, (x-hs, x+hs+1), (y-hs, y+hs+1))
        grid_feats_big = array.pad_slice(self.grid, (x-bhs, x+bhs+1), (y-bhs, y+bhs+1))
        grid_feats_red = block_reduce(grid_feats_big, (WINDOW_SIZE, WINDOW_SIZE, 1), func=np.max)

        pos_feats = np.asarray(self.pos, dtype=np.float32) / SIZE
        dir_feats = np.zeros(4)
        dir_feats[self.dir] = 1

        features = np.concatenate((
            grid_feats.ravel(),
            grid_feats_red.ravel(),
            self.inventory,
            dir_feats))
        self._cached_features = features
        return features

    def step(self, action):
        x, y = self.pos
        new_dir = self.dir
        new_inventory = self.inventory
        new_grid = self.grid
        reward = 0
        stop = False

        dx, dy = 0, 0
        if action == UP:
            dx, dy = 0, 1
            new_dir = UP
        elif action == DOWN:
            dx, dy = 0, -1
            new_dir = DOWN
        elif action == LEFT:
            dx, dy = -1, 0
            new_dir = LEFT
        elif action == RIGHT:
            dx, dy = 1, 0
            new_dir = RIGHT
        elif action == USE:
            for nx, ny in neighbors(self.pos, self.dir):
                here = self.grid[nx, ny, :]
                if not here.any():
                    continue
                if here.sum() > 1:
                    assert False
                assert here.sum() == 1
                thing = here.argmax()
                new_inventory = self.inventory.copy()
                new_grid = self.grid.copy()
                new_inventory[thing] += 1
                new_grid[nx, ny, thing] = 0
        elif action == CRAFT:
            for product, ingredients in self.world.recipes.items():
                if all(self.inventory[ing] > 0 for ing in ingredients):
                    new_inventory = self.inventory.copy()
                    new_inventory[product] += 1
                    for ing in ingredients:
                        new_inventory[ing] -= 1
                    break

        n_x = x + dx
        n_y = y + dy
        if n_x < 0 or n_x >= SIZE:
            n_x = x
        if n_y < 0 or n_y >= SIZE:
            n_y = y
        if self.grid[n_x, n_y, :].any():
            n_x, n_y = x, y

        if new_inventory[self.task.goal] > 0:
            reward = 1
            stop = True

        new_state = Minicraft2State(self.world, new_grid, (n_x, n_y), new_dir, new_inventory, self.task)
        return reward, new_state, stop
