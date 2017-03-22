from misc.util import Index

class SketchGuide(object):
    def __init__(self, config, world):
        self.max_len = world.max_hint_len
        self.n_vocab = len(world.vocab)

    def guide_for(self, task):
        return task.hint
