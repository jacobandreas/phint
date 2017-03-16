from misc.util import Index

class SketchGuide(object):
    def __init__(self, config):
        self.max_len = 6
        self.n_modules = 10
        self.index = Index()

    def guide_for(self, task):
        r = [self.index.index(s) for s in task.hint] + [0]
        assert len(r) <= self.max_len
        assert len(self.index) <= self.n_modules
        return r
