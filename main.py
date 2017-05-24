#!/usr/bin/env python2

from misc.util import Struct
import worlds
import models
from objectives import Reinforce, Ppo, Cloning
import trainers
from evaluators.zero_shot import ZeroShotEvaluator
from evaluators.adaptation import AdaptationEvaluator #, RllAdaptationEvaluator

import logging
import numpy as np
import os
import random
import sys
import tensorflow as tf
import traceback
import yaml

#from trainers import RlLabTrainer

def main():
    config = configure()
    world = worlds.load(config)

    # training pieces
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default(), session.as_default():
        model = models.load(config, world)
        if config.trainer.name == "ImitationTrainer":
            objective = Cloning(config, model)
        else:
            objective = Reinforce(config, model)
        trainer = trainers.load(config, session)
        #trainer = RlLabTrainer(config, world, model, session)

    # evaluation pieces
    zs_evaluator = ZeroShotEvaluator(config, world, model, session)

    eval_graph = tf.Graph()
    eval_session = tf.Session(graph=eval_graph)
    config_name = sys.argv[1]
    with open(config_name) as config_f:
        config_copy = Struct(**yaml.load(config_f))
    config_copy.model.controller.param_ling = False
    config_copy.model.controller.param_task = True
    with eval_graph.as_default(), eval_session.as_default():
        ad_model = models.load(config_copy, world)
        ad_objective = Reinforce(config_copy, ad_model)
        ad_evaluator = AdaptationEvaluator(config_copy, world, ad_model, ad_objective, eval_session)
        #ad_evaluator = RllAdaptationEvaluator(config_copy, world, ad_model, eval_session)
    def _evaluate_short():
        zs_evaluator.evaluate()

    def _evaluate_full():
        zs_evaluator.evaluate()
        with eval_graph.as_default(), eval_session.as_default():
            ad_evaluator.evaluate()

    # rllab might have screwed with this
    logging.getLogger().setLevel(logging.DEBUG)
    setup_logging(config)

    if config.train:
        with graph.as_default(), session.as_default():
            trainer.train(world, model, objective, _evaluate_short if config.eval else None)

    if config.eval:
        _evaluate_full()

def configure():
    # load config
    config_name = sys.argv[1]
    with open(config_name) as config_f:
        config = Struct(**yaml.load(config_f))

    # set up experiment
    config.experiment_dir = os.path.join("experiments/%s" % config.name)
    assert not os.path.exists(config.experiment_dir), \
            "Experiment %s already exists!" % config.experiment_dir
    os.mkdir(config.experiment_dir)

    return config

def setup_logging(config):
    log_name = os.path.join(config.experiment_dir, "run.log")
    logging.basicConfig(filename=log_name, level=logging.DEBUG,
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
            format='%(asctime)s %(levelname)-8s %(message)s')
    def handler(type, value, tb):
        logging.exception("Uncaught exception: %s", str(value))
        logging.exception("\n".join(traceback.format_exception(type, value, tb)))
    sys.excepthook = handler

    logging.info("BEGIN")
    logging.info(str(config))

if __name__ == "__main__":
    main()
