from curriculum import CurriculumTrainer

def load(config, session):
    cls_name = config.trainer.name
    try:
        cls = globals()[cls_name]
        return cls(config, session)
    except KeyError:
        raise Exception("No such trainer: {}".format(cls_name))
