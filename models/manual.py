import numpy as np
import readchar

class ManualModel(object):
    def __init__(self, config, world):
        self.random = np.random.RandomState(0)
        self.world = world

    def init(self, tasks):
        return [None]

    def act(self, states):
        c = readchar.readchar()
        if c == "w":
            a = self.world.forward
        elif c == "s":
            a = self.world.back
        elif c == "a":
            a = self.world.left
        elif c == "d":
            a = self.world.right
        elif c == "r":
            a = self.world.up
        elif c == "f":
            a = self.world.down
        elif c == "q":
            a = self.world.attack
        elif c == "e":
            a = self.world.use
        elif c == " ":
            a = self.world.jump
        elif c == "x":
            exit()
        else:
            a = self.world.stop
        a = self.world.actions.index(a)
        return [(a, None)]

    def experience(self, episode):
        pass

    def train(self):
        pass
