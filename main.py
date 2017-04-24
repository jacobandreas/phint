#!/usr/bin/env python2

from misc.util import Struct
import guides
import worlds
import models
from objectives import Reinforce, Ppo, Cloning
import trainers
from evaluators.zero_shot import ZeroShotEvaluator
from evaluators.adaptation import AdaptationEvaluator

import logging
import numpy as np
import os
import random
import sys
import tensorflow as tf
import traceback
import yaml

def main():
    config = configure()
    world = worlds.load(config)
    guide = guides.load(config, world)
    model = models.load(config, world, guide)
    objective = Reinforce(config, model)
    #objective = Cloning(config, model)

    session = tf.Session()

    def _evaluate():
        with open("config.yaml") as config_f:
            config_copy = Struct(**yaml.load(config_f))
        zs_evaluator = ZeroShotEvaluator(config_copy, session)
        zs_evaluator.evaluate(world, model)

        config_copy.model.controller.param_ling = False
        config_copy.model.controller.param_task = True
        ad_evaluator = AdaptationEvaluator(config_copy, session)
        ad_objective = Reinforce(config_copy, model)
        ad_evaluator.evaluate(world, model, ad_objective)

    if config.train:
        trainer = trainers.load(config, session)
        trainer.train(world, model, objective, _evaluate if config.eval else None)

    if config.eval:
        _evaluate()

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
