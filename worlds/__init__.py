import sys
try:
    from malmo import MalmoWorld
except Exception as e:
    print >>sys.stderr, "Warning: unable to load Malmo"
from minicraft import MinicraftWorld

def load(config):
    cls_name = config.world.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such world: {}".format(cls_name))
