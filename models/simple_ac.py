from misc import util
import net

from collections import namedtuple, defaultdict
import logging
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.framework.ops import IndexedSlicesValue

N_UPDATE = 2000
N_BATCH = 2000
#N_UPDATE = 500
#N_BATCH = 500

N_HIDDEN = 128
N_EMBED = 64

DISCOUNT = 0.95

ActorModule = namedtuple("ActorModule", ["t_probs", "t_chosen_prob", "params"])
CriticModule = namedtuple("CriticModule", ["t_value", "params"])
Trainer = namedtuple("Trainer", ["t_loss", "t_grad", "t_train_op"])
InputBundle = namedtuple("InputBundle", ["t_feats", "t_action_mask", "t_reward"])

ModelState = namedtuple("ModelState", ["action", "arg", "remaining", "task", "step"])

def increment_sparse_or_dense(into, increment):
    assert isinstance(into, np.ndarray)
    if isinstance(increment, IndexedSlicesValue):
        for i in range(increment.values.shape[0]):
            into[increment.indices[i], :] += increment.values[i, :]
    else:
        into += increment

class SimpleACModel(object):
    def __init__(self, config, world):
        self.experiences = []
        self.world = None
        tf.set_random_seed(0)
        self.next_actor_seed = 0
        self.config = config
        self.random = np.random.RandomState(0)
        self.prepare(world)

    def prepare(self, world):
        assert self.world is None
        self.world = world
        self.n_features = world.n_features

        self.n_actions = world.n_actions
        # TODO configurable optimizer
        self.optimizer = tf.train.AdamOptimizer(0.001)

        def build_actor(index, t_input, t_action_mask, extra_params=[]):
            with tf.variable_scope("actor_%s" % index):
                t_action_score, v_action = net.mlp(t_input, (N_HIDDEN, self.n_actions))
                #t_action_score, v_action = net.mlp(t_input, (self.n_actions,))

                t_action_logprobs = tf.nn.log_softmax(t_action_score)
                t_chosen_prob = tf.reduce_sum(t_action_mask * t_action_logprobs, 
                        reduction_indices=(1,))

            return ActorModule(t_action_logprobs, t_chosen_prob, 
                    v_action+extra_params)

        def build_critic(index, t_input, extra_params=[]):
            with tf.variable_scope("critic_%s" % index):
                if self.config.model.baseline in ("task", "common"):
                    t_value = tf.get_variable("b", shape=(),
                            initializer=tf.constant_initializer(0.0))
                    v_value = [t_value]
                elif self.config.model.baseline == "state":
                    t_value, v_value = net.mlp(t_input, (1,))
                    t_value = tf.squeeze(t_value)
                else:
                    raise NotImplementedError(
                            "Baseline %s is not implemented" % self.config.model.baseline)
            return CriticModule(t_value, v_value + extra_params)

        def build_actor_trainer(actor, critic, t_reward):
            t_advantage = t_reward - critic.t_value
            # TODO configurable entropy regularizer
            actor_loss = -tf.reduce_sum(actor.t_chosen_prob * t_advantage) + \
                    0.001 * tf.reduce_sum(tf.exp(actor.t_probs) * actor.t_probs)
            actor_grad = tf.gradients(actor_loss, actor.params)
            actor_trainer = Trainer(actor_loss, actor_grad, 
                    self.optimizer.minimize(actor_loss, var_list=actor.params))
            return actor_trainer

        def build_critic_trainer(critic, t_reward):
            t_advantage = t_reward - critic.t_value
            critic_loss = tf.reduce_sum(tf.square(t_advantage))
            critic_grad = tf.gradients(critic_loss, critic.params)
            critic_trainer = Trainer(critic_loss, critic_grad,
                    self.optimizer.minimize(critic_loss, var_list=critic.params))
            return critic_trainer

        # placeholders
        t_feats = tf.placeholder(tf.float32, shape=(None, self.n_features))
        t_action_mask = tf.placeholder(tf.float32, shape=(None, self.n_actions))
        t_reward = tf.placeholder(tf.float32, shape=(None,))
        t_input = t_feats

        xp = []
        self.actor = build_actor(0, t_input, t_action_mask, extra_params=xp)
        self.critic = build_critic(0, t_input, extra_params=xp)
        self.actor_trainer = build_actor_trainer(self.actor, self.critic, t_reward)
        self.critic_trainer = build_critic_trainer(self.critic, t_reward)

        self.t_gradient_placeholders = {}
        self.t_update_gradient_op = None

        params = self.actor.params + self.critic.params
        self.saver = tf.train.Saver()

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

        self.inputs = InputBundle(t_feats, t_action_mask, t_reward)

    def init(self):
        pass

    def save(self):
        self.saver.save(self.session, 
                os.path.join(self.config.experiment_dir, "modular_ac.chk"))

    def load(self):
        path = os.path.join(self.config.experiment_dir, "modular_ac.chk")
        logging.info("loaded %s", path)
        self.saver.restore(self.session, path)

    def experience(self, episode):
        running_reward = 0
        for transition in episode[::-1]:
            running_reward = running_reward * DISCOUNT + transition.r
            n_transition = transition._replace(r=running_reward)
            if n_transition.a < self.n_actions:
                self.experiences.append(n_transition)

    def act(self, obs):
        feed_dict = {
            self.inputs.t_feats: obs
        }
        lprobs = self.session.run([self.actor.t_probs], feed_dict=feed_dict)[0]
        probs = np.exp(lprobs)
        return [self.random.choice(self.n_actions, p=pr) for pr in probs]

    def train(self):
        experiences = self.experiences
        if len(experiences) < N_UPDATE:
            return None

        batch = experiences[:N_UPDATE]

        grads = {}
        params = {}
        for module in [self.actor, self.critic]:
            for param in module.params:
                if param.name not in grads:
                    grads[param.name] = np.zeros(param.get_shape(), np.float32)
                    params[param.name] = param

        total_actor_err = 0
        total_critic_err = 0
        total_reward = 0
        for i_batch in range(int(np.ceil(1. * len(experiences) / N_BATCH))):
            exps = experiences[i_batch * N_BATCH : (i_batch + 1) * N_BATCH]
            s1, m1, a, s2, m2, r = zip(*exps)
            a_mask = np.zeros((len(exps), self.n_actions))
            for i_datum, aa in enumerate(a):
                a_mask[i_datum, aa] = 1

            feed_dict = {
                self.inputs.t_feats: s1,
                self.inputs.t_action_mask: a_mask,
                self.inputs.t_reward: r
            }

            actor_grad, actor_err = self.session.run(
                    [self.actor_trainer.t_grad, self.actor_trainer.t_loss],
                    feed_dict=feed_dict)
            critic_grad, critic_err = self.session.run(
                    [self.critic_trainer.t_grad, self.critic_trainer.t_loss], 
                    feed_dict=feed_dict)

            total_actor_err += actor_err
            total_critic_err += critic_err
            total_reward += np.sum(r)

            for param, grad in zip(self.actor.params, actor_grad):
                increment_sparse_or_dense(grads[param.name], grad)

            for param, grad in zip(self.critic.params, critic_grad):
                increment_sparse_or_dense(grads[param.name], grad)

        global_norm = 0
        for k in params:
            grads[k] /= N_UPDATE
            global_norm += (grads[k] ** 2).sum()
        rescale = min(1., 1. / global_norm)

        # TODO precompute this part of the graph
        updates = []
        feed_dict = {}
        for k in params:
            param = params[k]
            grad = grads[k]
            grad *= rescale
            if k not in self.t_gradient_placeholders:
                self.t_gradient_placeholders[k] = tf.placeholder(tf.float32, grad.shape)
            feed_dict[self.t_gradient_placeholders[k]] = grad
            updates.append((self.t_gradient_placeholders[k], param))
        if self.t_update_gradient_op is None:
            self.t_update_gradient_op = self.optimizer.apply_gradients(updates)
        self.session.run(self.t_update_gradient_op, feed_dict=feed_dict)

        self.experiences = []

        return np.asarray([total_actor_err, total_critic_err, total_reward]) / N_UPDATE
