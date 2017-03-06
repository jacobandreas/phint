from misc import util

import bisect
from collections import defaultdict, namedtuple
import logging
import numpy as np
import tensorflow as tf

INIT_SCALE = 1.43

ActorState = namedtuple("ActorState", ["step"])
ControllerState = namedtuple("ControllerState", ["task", "hint", "obs", "index"])

class DiscreteDist(object):
    def __init__(self):
        self.random = util.next_random()

    def sample(self, param, temp):
        prob = np.exp(param)
        #prob = np.exp(param / np.exp(temp[:, np.newaxis]))
        prob /= prob.sum(axis=1, keepdims=True)
        return [self.random.choice(len(row), p=row) for row in prob]

    def log_prob_of(self, t_param, t_temp, t_action):
        #t_score = t_param / tf.exp(tf.expand_dims(t_temp, axis=1))
        t_score = t_param
        t_log_prob = tf.nn.log_softmax(t_score)
        t_chosen = util.batch_gather(t_log_prob, t_action)
        return t_chosen

    def entropy(self, t_param, t_temp):
        t_prob = tf.nn.softmax(t_param)
        t_logprob = tf.log(t_prob)
        return tf.reduce_sum(t_prob * t_logprob, axis=1)

class DiscreteActors(object):
    def __init__(self, config, t_obs, world, guide):
        with tf.variable_scope("actors"):
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
                    init[-1, :] = config.model.actor.ret_bias
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

        self.t_action_param = tf.slice(
                prev_layer,
                (0, 0, 0),
                (-1, world.n_act, -1))
        self.t_ret_param = tf.slice(
                prev_layer,
                (0, world.n_act, 0),
                (-1, -1, -1))

        self.t_action_temp = tf.get_variable(
                "action_temp",
                shape=(guide.n_modules,),
                initializer=tf.constant_initializer(0))
        self.t_ret_temp = tf.get_variable(
                "ret_temp",
                shape=(guide.n_modules,),
                initializer=tf.constant_initializer(0))

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
        with tf.variable_scope("critic"):
            v_w = tf.get_variable(
                    "w",
                    shape=(world.n_obs, world.n_tasks+1),
                    initializer=tf.uniform_unit_scaling_initializer(
                        factor=INIT_SCALE))
            v_b = tf.get_variable(
                    "b",
                    shape=(world.n_tasks+1),
                    initializer=tf.constant_initializer(0))
            self.t_value = tf.einsum("ij,jk->ik", t_obs, v_w) + v_b

class SketchController(object):
    def __init__(self, config, t_obs, world, guide):
        self.guide = guide
        self.task_index = util.Index()
        self.t_attention = tf.placeholder(tf.float32, shape=(None, guide.n_modules))
        self.t_task = tf.placeholder(tf.int32, shape=(None,))

    def init(self, task, obs):
        return [ControllerState(
                    self.task_index.index(t.mission), self.guide.guide_for(t),
                    None, 0)
                for t in task]

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

class ModularModel(object):
    def __init__(self, config, world, guide):
        self.world = world
        self.guide = guide
        self.config = config

        self.action_dist = DiscreteDist()

        self.prepare(config, world, guide)

    def prepare(self, config, world, guide):
        self.t_obs = tf.placeholder(tf.float32, (None, world.n_obs))

        if config.model.actor.type == "discrete":
            self.actors = DiscreteActors(config, self.t_obs, world, guide)
        else:
            assert False

        if config.model.controller.type == "sketch":
            self.controller = SketchController(config, self.t_obs, world, guide)
        else:
            assert False

        if config.model.critic.type == "linear":
            self.critic = LinearCritic(config, self.t_obs, world, guide)
        else:
            assert False

        t_att = self.controller.t_attention
        t_att_bc = tf.expand_dims(t_att, axis=1)
        self.t_action_param = tf.reduce_sum(
                self.actors.t_action_param * t_att_bc, axis=2)
        self.t_ret_param = tf.reduce_sum(
                self.actors.t_ret_param * t_att_bc, axis=2)
        self.t_action_temp = tf.reduce_sum(
                self.actors.t_action_temp * t_att, axis=1)
        self.t_ret_temp = tf.reduce_sum(
                self.actors.t_ret_temp * t_att, axis=1)
        self.t_baseline = util.batch_gather(self.critic.t_value, self.controller.t_task)

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
        action_p, ret_p, action_t, ret_t = session.run(
            [self.t_action_param, self.t_ret_param, self.t_action_temp, self.t_ret_temp],
            self.feed(obs, mstate))

        action = self.action_dist.sample(action_p, action_t)
        ret = self.action_dist.sample(ret_p, ret_t)
        any_ret = list(ret)

        actor_state_ = self.actors.step(actor_state, action, ret)

        for i in range(len(obs)):
            if actor_state_[i].step >= 15:
                any_ret[i] = 1
            if any_ret[i] == 1:
                actor_state_[i] = actor_state_[i]._replace(step=0)

        controller_state_, stop = self.controller.step(controller_state, action, any_ret)

        mstate_ = zip(actor_state_, controller_state_)

        return zip(action, ret), stop, mstate_
