import numpy as np
import tensorflow as tf

EPS = 0.2

class Ppo(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model

        self.t_action = tf.placeholder(tf.int32, (None,))
        self.t_ret = tf.placeholder(tf.int32, (None,))
        self.t_reward = tf.placeholder(tf.float32, (None,))

        t_lr = model.action_dist.likelihood_ratio_of(
                model.t_action_param, model.t_action_bias, model.t_action_temp,
                model.t_action_param_old, model.t_action_bias_old, model.t_action_temp_old,
                self.t_action, self.t_ret)
        t_lr_clip = tf.clip_by_value(t_lr, 1-EPS, 1+EPS)

        t_advantage = (self.t_reward - model.t_baseline)

        t_ppo = tf.minimum(t_lr * t_advantage, t_lr_clip * t_advantage)

        self.t_actor_loss = -tf.reduce_mean(
                t_ppo
                + config.objective.entropy_bonus
                    * model.action_dist.entropy(model.t_action_param, model.t_action_temp))

        self.t_critic_loss = tf.reduce_mean(
                tf.square(self.t_reward - model.t_baseline))

        optimizer = tf.train.RMSPropOptimizer(config.objective.step_size)
        self.o_train_actor = optimizer.minimize(self.t_actor_loss, var_list=model.actor_params)
        self.o_train_critic = optimizer.minimize(self.t_critic_loss, var_list=model.critic_params)

        self.buffer = []

    def experience(self, episodes):
        for episode in episodes:
            running_reward = 0
            for transition in episode[::-1]:
                running_reward = running_reward * self.config.objective.discount + transition.r
                n_transition = transition._replace(r=running_reward)
                self.buffer.append(n_transition)

    def ready(self):
        return len(self.buffer) >= self.config.objective.n_train_batch

    def train(self, session):
        for _ in range(5):
            self.buffer = self.buffer[:self.config.objective.n_train_batch]
            s1, m1, a, s2, m2, r = zip(*self.buffer)
            act, ret = zip(*a)
            feed = {
                self.t_action: act,
                self.t_ret: ret,
                self.t_reward: r,
            }
            feed.update(self.model.feed(s1, m1))

            actor_err, critic_err, _, _, b = session.run(
                    [self.t_actor_loss, self.t_critic_loss, 
                        self.o_train_actor, self.o_train_critic, self.model.t_baseline],
                    feed)
            session.run(self.model.oo_update_old)

        self.buffer = []
        return np.asarray([actor_err, critic_err])
