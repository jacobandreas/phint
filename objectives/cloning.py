import numpy as np
import tensorflow as tf

class Cloning(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model

        self.t_action = tf.placeholder(tf.int32, (None,))
        self.t_ret = None

        t_prob = model.action_dist.log_prob_of(
                model.t_action_param, model.t_action_bias, model.t_action_temp,
                self.t_action, self.t_ret)
        self.t_actor_loss = -tf.reduce_mean(t_prob)

        optimizer = tf.train.RMSPropOptimizer(config.objective.step_size)
        self.o_train_actor = optimizer.minimize(self.t_actor_loss)

        self.buffer = []

    def experience(self, episodes):
        for ep in episodes:
            self.buffer += ep

    def ready(self):
        return len(self.buffer) >= self.config.objective.n_train_batch

    def train(self, session):
        self.buffer = self.buffer[:self.config.objective.n_train_batch]
        s1, m1, a, s2, m2, r = zip(*self.buffer)
        #act, ret = zip(*a)
        act = a
        feed = {
            self.t_action: act
        }
        feed.update(self.model.feed(s1, m1))
        actor_err, _ = session.run([self.t_actor_loss, self.o_train_actor], feed)
        self.buffer = []
        return actor_err
