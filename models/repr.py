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
    def __init__(self, config, t_obs, t_task, world, guide):
        self.guide = guide

        self.t_hint = tf.placeholder(tf.int32, shape=(None, None))
        self.t_len = tf.placeholder(tf.int32, shape=(None,))
        #self.t_task = tf.placeholder(tf.int32, shape=(None,))

        with tf.variable_scope("ling_repr"):
            t_embed = _embed(self.t_hint, guide.n_vocab, config.model.controller.n_embed)
            cell = tf.contrib.rnn.GRUCell(config.model.controller.n_hidden)
            _, t_final_hidden = tf.nn.dynamic_rnn(cell, t_embed, self.t_len, dtype=tf.float32)
            t_ling_repr = t_final_hidden

        with tf.variable_scope("task_repr"):
            t_task_repr = _embed(
                    t_task, guide.n_tasks, config.model.controller.n_hidden)

        param_ling = config.model.controller.param_ling
        param_task = config.model.controller.param_task
        if param_ling and param_task:
            self.t_repr = t_ling_repr + t_task_repr
        elif param_ling:
            self.t_repr = t_ling_repr
        elif param_task:
            self.t_repr = t_task_repr
        else:
            self.t_repr = tf.zeros_like(t_task_repr)
        self.t_ling_repr = t_ling_repr

        with tf.variable_scope("ling_decoder"):
            cell = tf.contrib.rnn.GRUCell(config.model.controller.n_hidden)
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell, guide.n_vocab)
            t_dec_embed = t_embed[:, :-1, :]
            t_dec_target = self.t_hint[:, 1:]
            t_dec_pred, _ = tf.nn.dynamic_rnn(cell, t_dec_embed, initial_state=self.t_repr)
            self.t_dec_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=t_dec_target,
                        logits=t_dec_pred),
                    axis=1)

    def init(self, inst, obs):
        return [ModelState(i.task.id, self.guide.guide_for(i.task)) for i in inst]

    def feed(self, state):
        max_len = max(len(s.hint) for s in state)
        return {
            self.t_hint: [s.hint + (0,)*(max_len-len(s.hint)) for s in state],
            #self.t_task: [s.task_id for s in state],
            self.t_len: [len(s.hint) for i in state]
        }

class Actor(object):
    def __init__(self, config, t_obs, t_repr, world, guide):
        prev_layer = tf.concat((t_obs, t_repr), axis=1)
        #prev_layer = t_obs
        #widths = config.model.actor.n_hidden + [world.n_act + 2]
        widths = config.model.actor.n_hidden + [world.n_act]
        activations = [tf.nn.tanh] * (len(widths) - 1) + [None]
        with tf.variable_scope("actor") as scope:
            #bias = np.zeros((1, world.n_act + 2), dtype=np.float32)
            bias = np.zeros((1, world.n_act), dtype=np.float32)
            bias[0, -2:] = config.model.actor.ret_bias
            t_bias = tf.constant(bias)
            self.t_action_param = _mlp(prev_layer, widths, activations) + t_bias

class Critic(object):
    def __init__(self, config, t_obs, t_repr, t_task, world, guide):
        self.t_value = tf.squeeze(_linear(t_obs, 1))
        if config.model.critic.by_task:
            self.t_task_weight = _embed(t_task, guide.n_tasks, world.n_obs)
            self.t_value += tf.reduce_sum(self.t_task_weight * t_obs, axis=1)

class ReprModel(object):
    def __init__(self, config, world, guide):
        self.world = world
        self.guide = guide
        self.config = config
        self.prepare(config, world, guide)
        self.action_dist = DiscreteDist()
        self.saver = tf.train.Saver()

    def prepare(self, config, world, guide):
        self.t_obs = tf.placeholder(tf.float32, shape=(None, world.n_obs))
        self.t_task = tf.placeholder(tf.int32, shape=(None,))
        with tf.variable_scope("ReprModel") as scope:
            self.controller = EmbeddingController(config, self.t_obs, self.t_task, world, guide)
            self.actor = Actor(config, self.t_obs, self.controller.t_repr, world, guide)
            self.critic = Critic(config, self.t_obs, self.controller.t_repr, self.t_task, world, guide)
            self.params = util.vars_in_scope(scope)
        self.t_action_param = self.actor.t_action_param
        self.t_action_bias = 0
        self.t_action_temp = 0
        self.t_baseline = self.critic.t_value

        self.t_loss_extra = 0
        enc_ling = self.config.model.controller.encode_ling
        dec_ling = self.config.model.controller.decode_ling
        if enc_ling > 0:
            self.t_loss_extra += tf.reduce_mean(enc_ling * 
                tf.nn.l2_loss(self.controller.t_repr - self.controller.t_ling_repr))
        if dec_ling > 0:
            self.t_loss_extra += tf.reduce_mean(dec_ling * self.controller.t_dec_loss)

    def prepare_sym(self, obs_var, task_id_var):
        with tf.variable_scope("ReprModel", reuse=True) as scope:
            controller = EmbeddingController(self.config, obs_var, task_id_var, self.world, self.guide)
            actor = Actor(self.config, obs_var, controller.t_repr, self.world, self.guide)
        return tf.nn.softmax(actor.t_action_param)


    def init(self, task, obs):
        return self.controller.init(task, obs)

    def feed(self, obs, mstate):
        out = {
            self.t_obs: obs,
            self.t_task: [s.task_id for s in mstate]
        }
        out.update(self.controller.feed(mstate))
        return out

    def act(self, obs, mstate, task, session):
        n_obs = len(obs)
        action_p, = session.run([self.t_action_param], self.feed(obs, mstate))
        action, ret, _ = self.action_dist.sample(action_p, None, None)
        stop = [False] * n_obs
        return zip(action, ret), stop, mstate, [0]*n_obs

    def get_action_param(self, obs, mstate, task, session):
        n_obs = len(obs)
        action_p, rep = session.run([self.t_action_param, self.controller.t_repr], self.feed(obs, mstate))
        return action_p

    def save(self, session):
        self.saver.save(session, os.path.join(self.config.experiment_dir, "repr.chk"))

    def load(self, source, session):
        self.saver.restore(
                session, 
                os.path.join("experiments", source, "repr.chk"))
