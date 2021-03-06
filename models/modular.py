from misc import util
from net import _linear, _embed
from dists import DiscreteDist

import bisect
from collections import defaultdict, namedtuple
import logging
import numpy as np
import tensorflow as tf
import os

ActorState = namedtuple("ActorState", ["step"])
ControllerState = namedtuple(
        "ControllerState", 
        ["task", "hint", "obs", "index", "current_att", "total_att", "sampler_state"])

class DiscreteActors(object):
    def __init__(self, config, t_obs, world, guide):
        with tf.variable_scope("actors") as scope:
            prev_width = world.n_obs
            prev_layer = t_obs
            widths = config.model.actor.n_hidden + [world.n_act + 2]
            activations = [tf.nn.tanh] * (len(widths) - 1) + [None]
            for i_layer, (width, act) in enumerate(zip(widths, activations)):
                v_w = tf.get_variable(
                        "w_%d" % i_layer,
                        shape=(prev_width, width, guide.n_modules),
                        initializer=tf.uniform_unit_scaling_initializer(
                            factor=INIT_SCALE))
                if i_layer == len(widths) - 1:
                    init = np.zeros((width, guide.n_modules))
                    init[world.n_act, :] = config.model.actor.ret_bias
                else:
                    init = 0
                v_b = tf.get_variable(
                        "b_%d" % i_layer,
                        shape=(width, guide.n_modules),
                        initializer=tf.constant_initializer(init))
                if len(prev_layer.get_shape()) == 2:
                    op = "ij,jkl->ikl"
                else:
                    op = "ijl,jkl->ikl"
                layer = tf.einsum(op, prev_layer, v_w) + v_b
                if act is not None:
                    layer = act(layer)
                prev_layer = layer
                prev_width = width

                self.params = util.vars_in_scope(scope)

        #self.t_action_param = prev_layer
        #self.t_action_bias = v_b
        mask = np.zeros((world.n_act + 2, guide.n_modules), dtype=np.float32)
        mask[:-1, 0] = -10
        mask[-1, 1:] = -10
        self.t_action_bias = v_b
        self.t_action_param = tf.nn.softmax(layer + tf.constant(mask), dim=1)
        self.t_action_temp = tf.stop_gradient(tf.exp(tf.get_variable(
                "action_temp",
                shape=(guide.n_modules,),
                initializer=tf.constant_initializer(0))))

    def init(self, task, obs):
        return [ActorState(0) for _ in task]

    def step(self, state, action, ret):
        return [s._replace(step=s.step + 1) for s in state]

    def feed(self, mstate):
        return {}

class EmbeddedActors(object):
    pass

class LinearCritic(object):
    def __init__(self, config, t_obs, world, guide):
        with tf.variable_scope("critic") as scope:
            self.t_value = _linear(t_obs, world.n_tasks+1)
            self.params = util.vars_in_scope(scope)

class SketchController(object):
    def __init__(self, config, t_obs, world, guide):
        self.guide = guide
        self.task_index = util.Index()
        self.t_attention = tf.placeholder(tf.float32, shape=(None, guide.n_modules))
        self.t_task = tf.placeholder(tf.int32, shape=(None,))

    def init(self, inst, obs):
        return [ControllerState(
                    self.task_index.index(it.task), self.guide.guide_for(it.task),
                    None, 0)
                for it in inst]

    def feed(self, state):
        attention = np.zeros((len(state), self.guide.n_modules))
        for i, s in enumerate(state):
            if s.index >= len(s.hint):
                continue
            attention[i, s.hint[s.index]] = 1
        return {
            self.t_attention: attention,
            self.t_task: [s.task for s in state]
        }

    def step(self, state, action, ret):
        state_ = []
        stop = []
        for i in range(len(state)):
            if not ret[i]:
                state_.append(state[i])
                stop.append(False)
                continue
            s_ = state[i]._replace(index=state[i].index + 1)
            state_.append(s_)
            stop.append(s_.index >= len(s_.hint))
        return state_, stop
        # TODO max task len

class AttController(object):
    def __init__(self, config, t_world_obs, world, guide):
        self.config = config
        self.guide = guide
        self.world = world
        self.task_index = util.Index()
        n_hidden = config.model.controller.n_hidden
        n_embed = config.model.controller.n_embed
        self.t_obs = tf.placeholder(tf.float32, shape=(None, world.n_obs))
        self.t_hint = tf.placeholder(tf.int32, shape=(None, guide.max_len))
        t_batch_size = tf.shape(self.t_hint)[0]
        self.t_task = tf.placeholder(tf.int32, shape=(None,))
        self.t_len = tf.placeholder(tf.int32, shape=(None,))
        self.t_gumbel = tf.placeholder(tf.float32, shape=(None, guide.n_modules))

        # attention to hint
        t_embed = _embed(self.t_hint, guide.n_modules, n_embed)
        cell = tf.contrib.rnn.GRUCell(n_hidden)
        t_hint_states, _ = tf.nn.bidirectional_dynamic_rnn(
                cell, cell, t_embed, self.t_len, dtype=tf.float32)
        t_hint_repr = tf.reduce_mean(t_hint_states, axis=0)
        t_state_repr = tf.expand_dims(tf.nn.relu(_linear(self.t_obs, n_hidden)), axis=1)
        t_att_score = tf.reduce_sum(t_hint_repr * t_state_repr, axis=2)
        #t_hint_att = tf.nn.softmax(t_att_score)

        # attention to modules
        t_rows = tf.expand_dims(tf.range(t_batch_size), 1)
        t_rows_tile = tf.tile(t_rows, (1, guide.max_len))
        t_indices = tf.stack((t_rows_tile, self.t_hint), axis=2)
        t_scattered = tf.scatter_nd(t_indices, t_att_score, [t_batch_size, guide.n_modules])
        t_scattered = t_scattered + tf.constant([[-3]+[0]*(guide.n_modules-1)], dtype=tf.float32)
        if config.model.controller.gumbel:
            t_scattered += 0.1 * self.t_gumbel

        self.t_seq_attention = tf.nn.softmax(t_att_score)
        self.t_mod_attention = tf.nn.softmax(t_scattered)

    def init(self, inst, obs):
        out = []
        for it, ob in zip(inst, obs):
            guide = self.guide.guide_for(it.task)
            out.append(ControllerState(
                    self.task_index.index(it.task), guide,
                    ob, None, np.zeros(len(guide)), np.zeros(len(guide)),
                    np.random.gumbel(size=self.guide.n_modules)))
        return out

    def feed(self, state):
        return {
            self.t_obs: [s.obs for s in state],
            self.t_hint: [s.hint + [0] * (self.guide.max_len - len(s.hint)) for s in state],
            self.t_task: [s.task for s in state],
            self.t_len: [len(s.hint) for s in state],
            self.t_gumbel: [s.sampler_state for s in state]
        }

    def step(self, state, action, ret, obs, att):
        state_ = []
        stop = []
        reward = []
        for i in range(len(state)):
            if ret[i] or self.config.model.controller.update == "continuous":
                old_total_att = state[i].total_att
                att_here = att[i, :len(state[i].hint)]
                new_total_att = np.minimum(old_total_att + att_here, 1)
                extra = np.sum(new_total_att - old_total_att)
                reward.append(extra)
                rand = np.random.gumbel(size=self.guide.n_modules)
                s_ = state[i]._replace(obs=obs[i], current_att=att,
                        total_att=new_total_att, sampler_state=rand)
            else:
                s_ = state[i]
                reward.append(0)
            state_.append(s_)
            #stop.append(np.random.random() < att[i][0])
            stop.append(action[i] == self.world.n_act + 1)
        return state_, stop, reward

class FreeAttController(object):
    def __init__(self, config, t_world_obs, world, guide):
        self.config = config
        self.guide = guide
        self.world = world
        self.task_index = util.Index()
        n_hidden = config.model.controller.n_hidden
        n_embed = config.model.controller.n_embed
        self.t_obs = tf.placeholder(tf.float32, shape=(None, world.n_obs))
        self.t_hint = tf.placeholder(tf.int32, shape=(None, guide.max_len))
        t_batch_size = tf.shape(self.t_hint)[0]
        self.t_task = tf.placeholder(tf.int32, shape=(None,))
        self.t_len = tf.placeholder(tf.int32, shape=(None,))
        self.t_gumbel = tf.placeholder(tf.float32, shape=(None, guide.n_modules))

        # attention to hint
        t_embed = _embed(self.t_hint, guide.n_modules, n_embed)
        cell = tf.contrib.rnn.GRUCell(n_hidden)
        t_hint_states, _ = tf.nn.bidirectional_dynamic_rnn(
                cell, cell, t_embed, self.t_len, dtype=tf.float32)
        t_hint_repr = tf.reduce_mean(t_hint_states, axis=0)
        t_state_repr = tf.expand_dims(tf.nn.relu(_linear(self.t_obs, n_hidden)), axis=1)
        t_att_score = tf.reduce_sum(t_hint_repr * t_state_repr, axis=2)
        #t_hint_att = tf.nn.softmax(t_att_score)

        # attention to modules
        t_rows = tf.expand_dims(tf.range(t_batch_size), 1)
        t_rows_tile = tf.tile(t_rows, (1, guide.max_len))
        t_indices = tf.stack((t_rows_tile, self.t_hint), axis=2)
        t_scattered = tf.scatter_nd(t_indices, t_att_score, [t_batch_size, guide.n_modules])
        t_scattered = t_scattered + tf.constant([[-3]+[0]*(guide.n_modules-1)], dtype=tf.float32)
        if config.model.controller.gumbel:
            t_scattered += 0.1 * self.t_gumbel

        self.t_seq_attention = tf.nn.softmax(t_att_score)
        self.t_mod_attention = tf.nn.softmax(t_scattered)

    def init(self, inst, obs):
        out = []
        for it, ob in zip(inst, obs):
            guide = self.guide.guide_for(it.task)
            out.append(ControllerState(
                    self.task_index.index(it.task), guide,
                    ob, None, np.zeros(len(guide)), np.zeros(len(guide)),
                    np.random.gumbel(size=self.guide.n_modules)))
        return out

    def feed(self, state):
        return {
            self.t_obs: [s.obs for s in state],
            self.t_hint: [s.hint + [0] * (self.guide.max_len - len(s.hint)) for s in state],
            self.t_task: [s.task for s in state],
            self.t_len: [len(s.hint) for s in state],
            self.t_gumbel: [s.sampler_state for s in state]
        }

    def step(self, state, action, ret, obs, att):
        state_ = []
        stop = []
        reward = []
        for i in range(len(state)):
            if ret[i] or self.config.model.controller.update == "continuous":
                old_total_att = state[i].total_att
                att_here = att[i, :len(state[i].hint)]
                new_total_att = np.minimum(old_total_att + att_here, 1)
                extra = np.sum(new_total_att - old_total_att)
                reward.append(extra)
                rand = np.random.gumbel(size=self.guide.n_modules)
                s_ = state[i]._replace(obs=obs[i], current_att=att,
                        total_att=new_total_att, sampler_state=rand)
            else:
                s_ = state[i]
                reward.append(0)
            state_.append(s_)
            #stop.append(np.random.random() < att[i][0])
            stop.append(action[i] == self.world.n_act + 1)
        return state_, stop, reward

class ModularModel(object):
    def __init__(self, config, world, guide):
        self.world = world
        self.guide = guide
        self.config = config

        self.action_dist = DiscreteDist()

        self.prepare(config, world, guide)
        self.saver = tf.train.Saver()

    def prepare(self, config, world, guide):
        self.t_obs = tf.placeholder(tf.float32, (None, world.n_obs))

        if config.model.actor.type == "discrete":
            self.actors = DiscreteActors(config, self.t_obs, world, guide)
            with tf.variable_scope("old"):
                self.actors_old = DiscreteActors(config, self.t_obs, world, guide)
        elif config.model.actor.type == "embedded":
            self.actors = EmbeddedActors(config, self.t_obs, world, guide)
            with tf.variable_scope("old"):
                self.actors_old = EmbeddedActors(config, self.t_obs, world, guide)
        else:
            assert False

        if config.model.controller.type == "sketch":
            self.controller = SketchController(config, self.t_obs, world, guide)
        elif config.model.controller.type == "att":
            self.controller = AttController(config, self.t_obs, world, guide)
        else:
            assert False

        if config.model.critic.type == "linear":
            self.critic = LinearCritic(config, self.t_obs, world, guide)
        else:
            assert False

        t_att = self.controller.t_mod_attention
        t_att_bc = tf.expand_dims(t_att, axis=1)
        self.t_action_param = tf.log(tf.reduce_sum(
                self.actors.t_action_param * t_att_bc, axis=2))
        self.t_action_bias = tf.reduce_sum(
                self.actors.t_action_bias * t_att_bc, axis=2)
        self.t_action_temp = tf.reduce_sum(
                self.actors.t_action_temp * t_att, axis=1)

        #print self.actors.t_action_temp.get_shape()
        #print t_att.get_shape()
        #print self.t_action_temp.get_shape()
        #exit()

        # TODO cleanup?
        self.t_action_param_old = tf.reduce_sum(
                self.actors_old.t_action_param * t_att_bc, axis=2)
        self.t_action_bias_old = tf.reduce_sum(
                self.actors_old.t_action_bias * t_att_bc, axis=2)
        self.t_action_temp_old = tf.reduce_sum(
                self.actors_old.t_action_temp * t_att, axis=1)

        self.t_baseline = util.batch_gather(self.critic.t_value, self.controller.t_task)

        self.oo_update_old = [op.assign(p) for p, op 
                in zip(self.actors.params, self.actors_old.params)]

        self.actor_params = self.actors.params
        self.critic_params = self.critic.params

    def init(self, task, obs):
        actor_state = self.actors.init(task, obs)
        controller_state = self.controller.init(task, obs)
        return zip(actor_state, controller_state)

    def feed(self, obs, mstate):
        actor_state, controller_state = zip(*mstate)
        actor_feed = self.actors.feed(actor_state)
        controller_feed = self.controller.feed(controller_state)
        out = {self.t_obs: obs}
        out.update(actor_feed)
        out.update(controller_feed)
        assert set(out.keys()) == set(
                [self.t_obs] + actor_feed.keys() + controller_feed.keys())
        return out

    def act(self, obs, mstate, task, session):
        actor_state, controller_state = zip(*mstate)
        action_p, action_b, action_t, att, ap = session.run(
            [self.t_action_param, self.t_action_bias, self.t_action_temp, self.controller.t_seq_attention, self.actors.t_action_param],
            self.feed(obs, mstate))

        action, ret = self.action_dist.sample(action_p, action_b, action_t)
        any_ret = list(ret)

        actor_state_ = self.actors.step(actor_state, action, ret)

        for i in range(len(obs)):
            if actor_state_[i].step >= 15:
                any_ret[i] = 1
            if any_ret[i] == 1:
                actor_state_[i] = actor_state_[i]._replace(step=0)

        controller_state_, stop, controller_reward = self.controller.step(controller_state, action, any_ret, obs, att)

        mstate_ = zip(actor_state_, controller_state_)

        return zip(action, ret), stop, mstate_, controller_reward

    def save(self, session):
        self.saver.save(
                session,
                os.path.join(self.config.experiment_dir, "modular.chk"))
