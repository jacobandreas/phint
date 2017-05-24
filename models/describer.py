import net
from misc import util

import numpy as np
import tensorflow as tf

class Describer(object):
    def __init__(self, config, world, session):
        self.config = config
        self.world = world
        self.session = session
        self.random = util.next_random()

        with tf.variable_scope("Describer") as scope:
            cell = tf.contrib.rnn.GRUCell(128)
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell, world.n_vocab)

            t_hint = tf.placeholder(tf.int32, (None, world.max_hint_len+1))
            t_context = t_hint[:, :-1]
            t_target = t_hint[:, 1:]
            t_emb_context = net._embed(t_context, world.n_vocab, 64)
            t_preds, _ = tf.nn.dynamic_rnn(cell, t_emb_context, dtype=tf.float32, scope=scope)
            t_scores = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=t_preds, labels=t_target)
            t_loss = tf.reduce_mean(t_scores)

            optimizer = tf.train.AdamOptimizer(0.001)
            o_train = optimizer.minimize(t_loss)

            scope.reuse_variables()

            t_prev = tf.placeholder(tf.int32, (None,))
            t_hidden = tf.placeholder(tf.float32, (None, 128))
            t_emb_prev = net._embed(t_prev, world.n_vocab, 64)
            t_next, t_next_hidden = cell(t_emb_prev, t_hidden)

            self.cell = cell
            self.t_hint = t_hint
            self.t_preds = t_preds
            self.t_loss = t_loss
            self.o_train = o_train

            self.t_prev = t_prev
            self.t_hidden = t_hidden
            self.t_next = t_next
            self.t_next_hidden = t_next_hidden

            self.variables = util.vars_in_scope(scope)

    def train(self):
        n_batch = self.config.trainer.n_rollout_batch

        self.session.run(tf.variables_initializer(self.variables))
        for i_epoch in range(20):
            total_loss = 0
            for i_iter in range(10):
                inst = [self.world.sample_train() for _ in range(n_batch)]
                hint = np.zeros((n_batch, self.world.max_hint_len+1))
                for i, it in enumerate(inst):
                    hint[i, 1:len(it.task.hint)+1] = it.task.hint

                loss, _ = self.session.run(
                        [self.t_loss, self.o_train], {self.t_hint: hint})
                total_loss += loss
            print total_loss

    def sample(self, n):
        hidden = np.zeros((n, 128))
        preds = np.zeros((n, self.world.max_hint_len+1), dtype=np.int32)
        for t in range(self.world.max_hint_len):
            last = preds[:, t]
            pred, next_hidden = self.session.run(
                    [self.t_next, self.t_next_hidden],
                    {self.t_prev: last, self.t_hidden: hidden})
            hidden = next_hidden
            for i, p in enumerate(pred):
                p = np.exp(p / 10)
                p /= np.sum(p)
                word = self.random.choice(self.world.n_vocab, p=p)
                preds[i, t+1] = word

        sents = []
        for i, words in enumerate(preds):
            sent = []
            for j in range(self.world.max_hint_len):
                if preds[i, j+1] == 0:
                    break
                sent.append(preds[i, j+1])
            sents.append(tuple(sent))

        return sents
