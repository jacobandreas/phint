from misc.util import Struct
import guides
import worlds
import models
from objectives.reinforce import Reinforce
import trainers

import logging
import numpy as np
import os
import random
import sys
import traceback
import yaml

def main():
    config = configure()
    world = worlds.load(config)
    guide = guides.load(config)
    model = models.load(config, world, guide)
    objectives = Reinforce(config, model)
    trainer = trainers.load(config)
    trainer.train(world, model, objectives)

def configure():
    # load config
    with open("config.yaml") as config_f:
        config = Struct(**yaml.load(config_f))

    # set up experiment
    config.experiment_dir = os.path.join("experiments/%s" % config.name)
    assert not os.path.exists(config.experiment_dir), \
            "Experiment %s already exists!" % config.experiment_dir
    os.mkdir(config.experiment_dir)

    # set up logging
    log_name = os.path.join(config.experiment_dir, "run.log")
    #logging.basicConfig(filename=log_name, level=logging.DEBUG,
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
            format='%(asctime)s %(levelname)-8s %(message)s')
    def handler(type, value, tb):
        logging.exception("Uncaught exception: %s", str(value))
        logging.exception("\n".join(traceback.format_exception(type, value, tb)))
    sys.excepthook = handler

    logging.info("BEGIN")
    logging.info(str(config))

    return config

if __name__ == "__main__":
    main()
