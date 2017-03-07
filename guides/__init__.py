from sketch import SketchGuide

def load(config):
    cls_name = config.guide.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such world: {}".format(cls_name))