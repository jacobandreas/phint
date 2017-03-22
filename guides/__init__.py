from sketch import SketchGuide

def load(config, world):
    cls_name = config.guide.name
    try:
        cls = globals()[cls_name]
        return cls(config, world)
    except KeyError:
        raise Exception("No such world: {}".format(cls_name))
