import numpy as np
import readchar

from worlds.malmo.client_manager import *

class ManualModel(object):
    def __init__(self, config, world):
        self.random = np.random.RandomState(0)
        self.world = world

    def init(self, tasks):
        return [None]

    def act(self, states):
        c = readchar.readchar()
        if c == "w":
            a = FORWARD
        elif c == "s":
            a = BACK
        elif c == "a":
            a = LEFT
        elif c == "d":
            a = RIGHT
        elif c == "r":
            a = UP
        elif c == "f":
            a = DOWN
        elif c == "q":
            a = ATTACK
        elif c == "e":
            a = USE
        elif c == " ":
            a = JUMP
        elif c == "x":
            exit()
        else:
            a = self.world.stop
        a = ACTIONS.index(a)
        return [(a, None)]

    def experience(self, episode):
        pass

    def train(self):
        pass
