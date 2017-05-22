from curriculum import CurriculumTrainer
from imitation import ImitationTrainer
from demonstration import DemonstrationTrainer
#from rllab_wrapper import RlLabTrainer

def load(config, session):
    cls_name = config.trainer.name
    try:
        cls = globals()[cls_name]
        return cls(config, session)
    except KeyError as e:
        raise Exception("No such trainer: {}".format(cls_name))
