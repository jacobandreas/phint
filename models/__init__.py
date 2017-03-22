from modular import ModularModel
from repr import ReprModel

def load(config, *args):
    cls_name = config.model.name
    try:
        cls = globals()[cls_name]
        return cls(config, *args)
    except KeyError:
        raise Exception("No such model: {}".format(cls_name))
