from misc import util
from net import _linear, _embed, _mlp
from dists import DiscreteDist

from collections import namedtuple
import logging
import numpy as np
import tensorflow as tf
import os

ModelState = namedtuple("ModelState", ["task_id", "hint"])

class EmbeddingController(object):
    def __init__(self, config, t_obs, world, guide):
        self.guide = guide

        self.t_hint = tf.placeholder(tf.int32, shape=(None, guide.max_len))
        self.t_len = tf.placeholder(tf.int32, shape=(None,))
        self.t_task = tf.placeholder(tf.int32, shape=(None,))

        with tf.variable_scope("ling_repr"):
            t_embed = _embed(self.t_hint, guide.n_vocab, config.model.controller.n_embed)
            cell = tf.contrib.rnn.GRUCell(config.model.controller.n_hidden)
            _, t_final_hidden = tf.nn.dynamic_rnn(cell, t_embed, self.t_len, dtype=tf.float32)
            t_ling_repr = t_final_hidden

        with tf.variable_scope("task_repr"):
            t_task_repr = _embed(
                    self.t_task, guide.n_tasks, config.model.controller.n_hidden)

        use_ling = config.model.controller.use_ling
        use_task = config.model.controller.use_task
        if use_ling and use_task:
            self.t_repr = t_ling_repr + t_task_repr
        elif use_ling:
            self.t_repr = t_ling_repr
        elif use_task:
            self.t_repr = t_task_repr
        else:
            self.t_repr = tf.zeros_like(t_task_repr)

    def init(self, inst, obs):
        return [ModelState(i.task.id, self.guide.guide_for(i.task)) for i in inst]

    def feed(self, state):
        return {
            self.t_hint: [s.hint + (0,)*(self.guide.max_len-len(s.hint)) for s in state],
            self.t_task: [s.task_id for s in state],
            self.t_len: [len(s.hint) for i in state]
        }

class Actor(object):
    def __init__(self, config, t_obs, t_repr, world, guide):
        prev_layer = tf.concat((t_obs, t_repr), axis=1)
        #widths = config.model.actor.n_hidden + [world.n_act + 2]
        widths = config.model.actor.n_hidden + [world.n_act]
        activations = [tf.nn.tanh] * (len(widths) - 1) + [None]
        with tf.variable_scope("actor") as scope:
            #bias = np.zeros((1, world.n_act + 2), dtype=np.float32)
            bias = np.zeros((1, world.n_act), dtype=np.float32)
            bias[0, -2:] = config.model.actor.ret_bias
            t_bias = tf.constant(bias)
            self.t_action_param = _mlp(prev_layer, widths, activations) + t_bias
            self.params = util.vars_in_scope

class Critic(object):
    def __init__(self, config, t_obs, t_repr, t_task, world, guide):
        if config.model.critic.by_task:
            self.t_weight = _embed(t_task, guide.n_tasks, world.n_obs)
            self.t_value = tf.reduce_sum(self.t_weight * t_obs, axis=1)
        else:
            self.t_value = tf.squeeze(_linear(t_obs, 1))

class ReprModel(object):
    def __init__(self, config, world, guide):
        self.world = world
        self.guide = guide
        self.config = config
        self.prepare(config, world, guide)
        self.saver = tf.train.Saver()
        self.action_dist = DiscreteDist()

    def prepare(self, config, world, guide):
        self.t_obs = tf.placeholder(tf.float32, (None, world.n_obs))
        self.controller = EmbeddingController(config, self.t_obs, world, guide)
        self.actor = Actor(config, self.t_obs, self.controller.t_repr, world, guide)
        self.critic = Critic(config, self.t_obs, self.controller.t_repr, self.controller.t_task, world, guide)
        self.t_action_param = self.actor.t_action_param
        self.t_action_bias = 0
        self.t_action_temp = 0
        self.t_baseline = self.critic.t_value

    def init(self, task, obs):
        controller_state = self.controller.init(task, obs)
        return controller_state

    def feed(self, obs, mstate):
        controller_state = mstate
        out = {self.t_obs: obs}
        out.update(self.controller.feed(controller_state))
        return out

    def act(self, obs, mstate, task, session):
        n_obs = len(obs)
        action_p, = session.run([self.t_action_param], self.feed(obs, mstate))
        action, ret, _ = self.action_dist.sample(action_p, None, None)
        stop = [False] * n_obs
        return zip(action, ret), stop, mstate, [0]*n_obs

    def save(self, session):
        self.saver.save(session, os.path.join(self.config.experiment_dir, "repr.chk"))

    def load(self, source, session):
        self.saver.restore(
                session, 
                os.path.join("experiments", source, "repr.chk"))
