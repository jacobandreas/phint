from misc import util

import tensorflow as tf
import numpy as np

TINY = 1e-8

class DiscreteDist(object):
    def __init__(self):
        self.random = util.next_random()

    def sample(self, param, bias, temp):
        #prob = np.exp(param * temp[:, np.newaxis] + bias)
        prob = np.exp(param * temp)
        #prob = np.exp(param)
        prob /= prob.sum(axis=1, keepdims=True)
        #print prob[0, :]
        #print "temp", temp
        actions = [self.random.choice(len(row), p=(row/np.sum(row))) for row in prob]
        n_actions = prob.shape[1]
        rets = [action == n_actions - 2 for action in actions]
        stops = [action == n_actions - 1 for action in actions]
        return actions, rets, stops

    def log_prob_of(self, t_param, t_bias, t_temp, t_action, t_ret):
        ##t_score = t_param * tf.expand_dims(t_temp, axis=1) + t_bias
        t_score = t_param * t_temp
        #t_score = t_param
        #t_log_prob = tf.nn.log_softmax(t_score)
        #t_chosen = util.batch_gather(t_log_prob, t_action)
        #return t_chosen
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=t_score, labels=t_action)

    def likelihood_ratio_of(self, t_param, t_bias, t_temp, t_param_old, t_bias_old, t_temp_old, t_action, t_ret):
        #t_score = t_param * tf.expand_dims(t_temp, axis=1) + t_bias
        #t_score_old = t_param_old * tf.expand_dims(t_temp_old, axis=1) + t_bias_old
        assert False # TODO temp
        t_score = t_param
        t_score_old = t_param_old
        t_prob = tf.nn.softmax(t_score)
        t_prob_old = tf.nn.softmax(t_score_old)
        t_chosen = util.batch_gather(t_prob, t_action)
        t_chosen_old = util.batch_gather(t_prob_old, t_action)
        return (t_chosen + TINY) / (t_chosen_old + TINY)

    def entropy(self, t_param, t_bias, t_temp):
        #t_prob = tf.nn.softmax(t_param * tf.expand_dims(t_temp, axis=1) + t_bias)
        t_prob = tf.nn.softmax(t_param * t_temp)
        #t_prob = tf.nn.softmax(t_param)
        t_logprob = tf.log(t_prob + TINY)
        return -tf.reduce_sum(t_prob * t_logprob, axis=1)

class DiagonalGaussianDist(object):
    def __init__(self):
        self.random = util.next_random()

    def sample(self, param, bias, temp):
        assert False # TODO temp is scalar
        cov = np.diag(np.exp(temp))
        actions = [self.random.multivariate_normal(param[i, :], cov)
                for i in range(param.shape[0])]
        rets = [False] * len(actions)
        stops = [False] * len(actions)
        return actions, rets, stops

    def log_prob_of(self, t_param, t_bias, t_temp, t_action, t_ret):
        return
