from misc import util
from net import _linear, _embed, _embed_pretrained, _mlp
from dists import DiscreteDist, DiagonalGaussianDist
import embeddings

from collections import namedtuple
import logging
import numpy as np
import tensorflow as tf
import os

ModelState = namedtuple("ModelState", ["task_id", "hint"])

#class EmbeddingController(object):
#    def __init__(self, config, t_obs, t_task, t_hint, world):
#        param_ling = config.model.controller.param_ling
#        param_task = config.model.controller.param_task
#
#        with tf.variable_scope("ling_repr") as repr_scope:
#            if hasattr(config.model.controller, "embeddings"):
#                embedding_dict = embeddings.load(
#                        config.model.controller.embeddings, world.vocab)
#                assert embedding_dict.shape[1] == config.model.controller.n_embed
#                t_embed = _embed_pretrained(t_hint, embedding_dict)
#            else:
#                t_embed = _embed(t_hint, world.n_vocab, config.model.controller.n_embed)
#            if not config.model.controller.train_embeddings:
#                t_embed = tf.stop_gradient(t_embed)
#            t_ling_repr = tf.reduce_mean(t_embed, axis=1)
#            self.repr_params = util.vars_in_scope(repr_scope)
#
#        with tf.variable_scope("task_repr") as repr_scope:
#            t_task_repr = _embed(
#                    t_task, world.n_tasks, config.model.controller.n_embed)
#            self.repr_params = util.vars_in_scope(repr_scope)
#
#        if param_ling and param_task:
#            self.t_repr = t_ling_repr + t_task_repr
#        elif param_ling:
#            self.t_repr = t_ling_repr
#        elif param_task:
#            self.t_repr = t_task_repr
#        else:
#            self.t_repr = tf.zeros_like(t_task_repr)
#        self.t_ling_repr = t_ling_repr
#
#        #self.t_repr = self.t_repr # + tf.random_normal(shape=tf.shape(self.t_repr), stddev=0.5)
#        self.t_repr = tf.nn.dropout(self.t_repr, 0.75)
#
#        self.t_dec_loss = 0
#        # TODO(jda) including this causes segfaults on the server
#        #with tf.variable_scope("ling_decoder"):
#        #    cell = tf.contrib.rnn.GRUCell(config.model.controller.n_hidden)
#        #    cell = tf.contrib.rnn.OutputProjectionWrapper(cell, world.n_vocab)
#        #    t_dec_embed = t_embed[:, :-1, :]
#        #    t_dec_target = self.t_hint[:, 1:]
#        #    t_dec_pred, _ = tf.nn.dynamic_rnn(cell, t_dec_embed, initial_state=self.t_repr)
#        #    self.t_dec_loss = tf.reduce_sum(
#        #            tf.nn.sparse_softmax_cross_entropy_with_logits(
#        #                labels=t_dec_target,
#        #                logits=t_dec_pred),
#        #            axis=1)
#
#    def init(self, inst, obs):
#        return [ModelState(i.task.id, i.task.hint) for i in inst]

class EmbeddingController(object):
    def __init__(self, config, t_obs, t_task, t_hint, world):
        param_ling = config.model.controller.param_ling
        param_task = config.model.controller.param_task

        param_size = config.model.actor.n_hidden[-1] * world.n_act

        with tf.variable_scope("ling_repr") as repr_scope:
            t_embed = _embed(t_hint, world.n_vocab, param_size)
            if not config.model.controller.train_embeddings:
                t_embed = tf.stop_gradient(t_embed)
            t_ling_repr = tf.reduce_mean(t_embed, axis=1)
            ling_params = util.vars_in_scope(repr_scope)

        with tf.variable_scope("task_repr") as repr_scope:
            t_task_repr = _embed(t_task, world.n_tasks, param_size)
            task_params = util.vars_in_scope(repr_scope)

        if not (param_ling or param_task):
            with tf.variable_scope("null_repr") as repr_scope:
                t_null_repr = tf.get_variable(
                        "repr",
                        shape=(param_size,),
                        initializer=tf.uniform_unit_scaling_initializer())
                t_null_repr = tf.expand_dims(t_null_repr, 0)
                t_null_repr = tf.tile(t_null_repr, (tf.shape(t_obs)[0], 1))
                null_params = util.vars_in_scope(repr_scope)
        else:
            null_params = []
            t_null_repr = 0

        if param_ling and param_task:
            self.t_repr = t_ling_repr + t_task_repr
        elif param_ling:
            self.t_repr = t_ling_repr
        elif param_task:
            self.t_repr = t_task_repr
        else:
            #self.t_repr = tf.zeros_like(t_task_repr)
            self.t_repr = t_null_repr
        self.t_ling_repr = t_ling_repr

        self.repr_params = ling_params + task_params + null_params

        # TODO keep?
        # TODO bias?
        #self.t_repr = tf.nn.dropout(self.t_repr, 0.75)

    def init(self, inst, obs):
        return [ModelState(i.task.id, i.task.hint) for i in inst]

#class Actor(object):
#    def __init__(self, config, t_obs, t_repr, world):
#        prev_layer = tf.concat((t_obs, t_repr), axis=1)
#        #prev_layer = t_obs
#        #widths = config.model.actor.n_hidden + [world.n_act + 2]
#        widths = config.model.actor.n_hidden + [world.n_act]
#        activations = [tf.nn.tanh] * (len(widths) - 1) + [None]
#        with tf.variable_scope("actor") as scope:
#            #bias = np.zeros((1, world.n_act + 2), dtype=np.float32)
#            bias = np.zeros((1, world.n_act), dtype=np.float32)
#            bias[0, -2:] = config.model.actor.ret_bias
#            t_bias = tf.constant(bias)
#            self.t_action_param = _mlp(prev_layer, widths, activations) + t_bias

class Actor(object):
    def __init__(self, config, t_obs, t_repr, world):
        widths = config.model.actor.n_hidden
        # TODO relu?
        activations = [tf.nn.tanh] * len(widths)
        with tf.variable_scope("actor") as scope:
            bias = np.zeros((1, world.n_act), dtype=np.float32)
            bias[0, -2:] = config.model.actor.ret_bias
            t_bias = tf.constant(bias)
            last_hidden = _mlp(t_obs, widths, activations)
            last_hidden = tf.nn.l2_normalize(last_hidden, 1)
            t_mat = tf.reshape(t_repr, (-1, widths[-1], world.n_act))
            self.t_action_param = tf.einsum("ij,ijk->ik", last_hidden, t_mat) + t_bias
            #self.t_action_param = tf.dot(t_repr, last_hidden) + t_bias

class Critic(object):
    def __init__(self, config, t_obs, t_repr, t_task, world):
        self.t_value = tf.squeeze(_linear(t_obs, 1))
        if config.model.critic.by_task:
            self.t_task_weight = _embed(t_task, world.n_tasks, world.n_obs)
            self.t_value += tf.reduce_sum(self.t_task_weight * t_obs, axis=1)

class ReprModel(object):
    def __init__(self, config, world):
        self.world = world
        self.config = config
        self.prepare(config, world)
        if world.is_discrete:
            self.action_dist = DiscreteDist()
        else:
            # TODO
            self.action_dist = DiscreteDist()
            #self.action_dist = DiagonalGaussianDist()
        self.saver = tf.train.Saver()

    def prepare(self, config, world):
        self.t_obs = tf.placeholder(tf.float32, shape=(None, world.n_obs))
        self.t_task = tf.placeholder(tf.int32, shape=(None,))
        self.t_hint = tf.placeholder(tf.int32, shape=(None, None))
        with tf.variable_scope("ReprModel") as scope:
            self.controller = EmbeddingController(
                    config, self.t_obs, self.t_task, self.t_hint, world)
            self.actor = Actor(
                    config, self.t_obs, self.controller.t_repr, world)
            self.critic = Critic(
                    config, self.t_obs, self.controller.t_repr, self.t_task,
                    world)

            self.t_action_param = self.actor.t_action_param
            self.t_action_temp = tf.get_variable(
                    #"temp", shape=(world.n_act,),
                    "temp", shape=(1),
                    #initializer=tf.constant_initializer(-1))
                    initializer=tf.constant_initializer(0))
            self.o_reset_temp = tf.assign(self.t_action_temp, (-1,))
            self.t_action_bias = 0

            self.params = util.vars_in_scope(scope)
            self.repr_params = self.controller.repr_params + [self.t_action_temp]

        self.t_baseline = self.critic.t_value

        self.t_loss_extra = 0
        if hasattr(self.config.model.controller, "encode_ling"):
            enc_ling = self.config.model.controller.encode_ling
            self.t_loss_extra += tf.reduce_mean(enc_ling * 
                tf.nn.l2_loss(self.controller.t_repr - self.controller.t_ling_repr))
        if hasattr(self.config.model.controller, "decode_ling"):
            dec_ling = self.config.model.controller.decode_ling
            self.t_loss_extra += tf.reduce_mean(dec_ling * self.controller.t_dec_loss)

    def prepare_sym(self, obs_var, task_id_var, hint_var):
        with tf.variable_scope("ReprModel", reuse=True) as scope:
            controller = EmbeddingController(
                    self.config, obs_var, task_id_var, hint_var, self.world)
            actor = Actor(
                    self.config, obs_var, controller.t_repr, self.world)
        return actor.t_action_param, self.t_action_temp

    def init(self, task, obs):
        return self.controller.init(task, obs)

    def feed(self, obs, mstate):
        max_len = self.world.max_hint_len
        out = {
            self.t_obs: obs,
            self.t_task: [s.task_id for s in mstate],
            self.t_hint: [s.hint + (0,)*(max_len-len(s.hint)) for s in mstate],
        }
        return out

    def act(self, obs, mstate, task, session):
        n_obs = len(obs)
        action_p, action_t = session.run(
                [self.t_action_param, self.t_action_temp],
                self.feed(obs, mstate))
        action, ret, _ = self.action_dist.sample(action_p, None, action_t)
        stop = [False] * n_obs
        return zip(action, ret), stop, mstate, [0]*n_obs

    def get_action(self, obs, mstate, task, session):
        n_obs = len(obs)
        action_p, action_t, rep = session.run(
                [self.t_action_param, self.t_action_temp, self.controller.t_repr], 
                self.feed(obs, mstate))
        return action_p, action_t, mstate

    def save(self, session):
        self.saver.save(session, os.path.join(self.config.experiment_dir, "repr.chk"))

    def load(self, source, session):
        self.saver.restore(
                session, 
                os.path.join("experiments", source, "repr.chk"))
