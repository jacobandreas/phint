#from manual import ManualModel
#from simple_ac import SimpleACModel
#from simple_q import SimpleQModel
#from recurrent_q import RecurrentQModel
#from memory import MemoryModel
from modular import ModularModel

def load(config, world):
    cls_name = config.model.name
    try:
        cls = globals()[cls_name]
        return cls(config, world)
    except KeyError:
        raise Exception("No such model: {}".format(cls_name))
