from misc import util
import net

import bisect
from collections import defaultdict, namedtuple
import logging
import numpy as np
import tensorflow as tf

N_BAD_BATCH = 100
N_RAND_BATCH = 100
N_GOOD_BATCH = 100
N_BATCH = N_BAD_BATCH + N_RAND_BATCH + N_GOOD_BATCH
N_STEPS_PER_UPDATE = 100
N_EXPERIENCES = 10000
N_UPDATE = 1000

N_HISTORY = 10
N_HINT = 5

N_RECURRENT = 64
N_HIDDEN = 64
N_EMBED = 64

N_TASKS = 100

DISCOUNT = 0.9

InputBundle = namedtuple("InputBundle", ["t_hint", "t_init_obs", "t_obs",
    "t_obs_target", "t_action_mask", "t_reward"])
QBundle = namedtuple("QBundle", ["t_q", "t_q_chosen", "t_q_best", "t_attention", "vars"])
ModelState = namedtuple("ModelState", ["hint", "init_obs", "index"])

class ModularModel(object):
    def __init__(self, config, world):
        self.experiences = []
        self.good_experiences = []
        self.bad_experiences = []
        tf.set_random_seed(0)
        self.config = config
        self.random = np.random.RandomState(0)
        self.prepare(world)

    def prepare(self, world):
        self.world = world
        self.n_features = world.n_features

        #self.attend_hint_action = world.n_actions
        #self.next_hint_action = world.n_actions + 1
        #self.attend_history_action = world.n_actions + 2
        #self.n_actions = self.attend_hint_action + 1

        self.next_hint_action = world.n_actions + 1
        self.n_actions = self.next_hint_action + 1

        self.optimizer = tf.train.AdamOptimizer(0.001)
        #self.optimizer = tf.train.AdamOptimizer(0.003)

        # shared
        t_hint = tf.placeholder(tf.int32, shape=(None, N_HINT))

        # one step
        t_init_obs = tf.placeholder(tf.float32, shape=(None, self.n_features))
        t_obs = tf.placeholder(tf.float32, shape=(None, self.n_features))
        t_obs_target = tf.placeholder(tf.float32, shape=(None, self.n_features))
        t_action_mask = tf.placeholder(tf.float32, shape=(None, self.n_actions))
        t_reward = tf.placeholder(tf.float32, shape=(None,))

        def build_net(name, t_hint, t_obs, t_action_mask):
            with tf.variable_scope(name) as scope:
                t_embed_hint, _ = net.embed(t_hint, N_TASKS, N_EMBED)

                with tf.variable_scope("repr") as repr_scope:
                    t_init_hidden, _ = net.mlp(t_init_obs, (N_HIDDEN,),
                            final_nonlinearity=True)
                    repr_scope.reuse_variables()
                    t_hidden, _ = net.mlp(t_obs, (N_HIDDEN,),
                            final_nonlinearity=True)
                with tf.variable_scope("att"):
                    t_attention_key, _ = net.mlp(t_init_hidden, (N_EMBED,))
                t_attention_rs = tf.reshape(t_attention_key, (-1, 1, N_HIDDEN))
                t_attention_tile = tf.tile(t_attention_rs, (1, N_HINT, 1))
                t_attention_mul = t_attention_tile * t_embed_hint
                t_attention_unnorm = tf.exp(tf.reduce_sum(t_attention_mul,
                        reduction_indices=(2,), keep_dims=True))
                t_normalizer = tf.reshape(tf.reduce_sum(t_attention_unnorm,
                        reduction_indices=(1, 2)), (-1, 1, 1))
                t_attention = t_attention_unnorm / t_normalizer
                t_hint_weighted = t_embed_hint * t_attention
                t_hint_repr = tf.reduce_sum(t_hint_weighted,
                        reduction_indices=(1,))

                t_concat = tf.concat(1, (t_hidden, t_hint_repr))
                with tf.variable_scope("act"):
                    t_q, _ = net.mlp(t_concat, (N_HIDDEN, self.n_actions))

                t_q_chosen = None
                if t_action_mask is not None:
                    t_q_chosen = tf.reduce_sum(t_action_mask * t_q,
                            reduction_indices=(1,))
                t_q_best = tf.reduce_max(t_q, reduction_indices=(1,))

                vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.name)

                return QBundle(t_q, t_q_chosen, t_q_best, t_attention, vars)

        qnet = build_net("net", t_hint, t_obs, t_action_mask)
        qnet_target = build_net("net_target", t_hint, t_obs_target, None)

        t_td = t_reward + DISCOUNT * qnet_target.t_q_best - qnet.t_q_chosen
        t_err = tf.reduce_mean(tf.square(t_td))

        t_train_op = self.optimizer.minimize(t_err, var_list=qnet.vars)
        t_update_ops = [v_target.assign(v) for v, v_target in zip(qnet.vars, qnet_target.vars)]

        self.qnet = qnet
        self.qnet_target = qnet_target
        self.t_err = t_err
        self.t_train_op = t_train_op
        self.t_update_ops = t_update_ops
        self.inputs = InputBundle(t_hint, t_init_obs, t_obs, t_obs_target,
                t_action_mask, t_reward)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

        self.i_step = 0
        self.i_task_step = defaultdict(lambda: 0)
        self.i_update = 0

        self.task_index = util.Index()

    def init(self, hints):
        embedded = [tuple(self.task_index.index(subtask) for subtask in hint) for
                hint in hints]
        self.mstates = [ModelState(e, np.zeros((self.n_features,)), 0) for e in embedded]
        return self.mstates

    def experience(self, episode):
        self.i_task_step[episode[0].m1.hint] += 1

        self.experiences += episode
        self.good_experiences += episode
        self.bad_experiences += episode

        self.random.shuffle(self.good_experiences)
        self.random.shuffle(self.bad_experiences)

        self.experiences = self.experiences[-N_EXPERIENCES:]
        self.good_experiences = list(
                sorted(self.good_experiences, key=lambda x: -x.r)
            )[:N_GOOD_BATCH]
        self.bad_experiences = list(
                sorted(self.bad_experiences, key=lambda x: x.r)
            )[:N_BAD_BATCH]

    def act(self, obs):
        hints = np.zeros((len(obs), N_HINT))
        init_obs = np.zeros((len(obs), self.n_features))
        attention = np.zeros((len(obs), N_HINT, 1))
        for i, m in enumerate(self.mstates):
            hints[i, :len(m.hint)] = m.hint
            init_obs[i, :] = m.init_obs
            attention[i, min(m.index, N_HINT-1), 0] = 1
        feed_dict = {
            self.inputs.t_hint: hints,
            #self.inputs.t_init_obs: init_obs,
            self.qnet.t_attention: attention,
            self.inputs.t_obs: np.asarray(obs),
        }
        q, = self.session.run([self.qnet.t_q], feed_dict=feed_dict)
        out = []
        for i_state in range(len(obs)):
            hint = self.mstates[i_state].hint
            if self.random.rand() < max(0.1, (10000 - self.i_task_step[hint]) / 10000.):
                action = self.random.randint(self.n_actions)
            else:
                action = np.argmax(q[i_state, :])
            #if action == self.attend_hint_action:
            #    self.mstates[i_state] = \
            #        self.mstates[i_state]._replace(init_obs=obs[i])
            if action == self.next_hint_action:
                old_state = self.mstates[i_state]
                self.mstates[i_state] = \
                        old_state._replace(index=old_state.index + 1)
            out.append(action)
        return out, self.mstates

    def train(self):
        logging.info("[train] %d", self.i_step)

        if len(self.experiences) < N_BATCH:
            return None

        total_err = 0
        for i_step in range(N_STEPS_PER_UPDATE):
            self.i_step += 1
            rand_indices = [self.random.randint(len(self.experiences)) for _ in range(N_RAND_BATCH)]
            rand_batch = [self.experiences[i] for i in rand_indices]
            batch = rand_batch + self.good_experiences + self.bad_experiences

            states0 = np.zeros((N_BATCH, self.n_features))
            states1 = np.zeros((N_BATCH, self.n_features))
            states2 = np.zeros((N_BATCH, self.n_features))
            action_masks = np.zeros((N_BATCH, self.n_actions))
            rewards = np.zeros((N_BATCH,))
            hints = np.zeros((N_BATCH, N_HINT))
            for i_experience in range(N_BATCH):
                exp = batch[i_experience]
                states0[i_experience, :] = exp.m1.init_obs
                states1[i_experience, :] = exp.s1
                states2[i_experience, :] = exp.s2
                action_masks[i_experience, exp.a] = 1
                rewards[i_experience] = exp.r
                hints[i_experience, :len(exp.m1.hint)] = exp.m1.hint
            feed_dict = {
                self.inputs.t_hint: hints,
                self.inputs.t_init_obs: states0,
                self.inputs.t_obs: states1,
                self.inputs.t_obs_target: states2,
                self.inputs.t_action_mask: action_masks,
                self.inputs.t_reward: rewards,
            }
            err, _ = self.session.run([self.t_err, self.t_train_op], feed_dict=feed_dict)
            total_err += err

        if self.i_step >= N_UPDATE:
            logging.info("[update] %d", self.i_update)
            self.session.run(self.t_update_ops)
            self.i_step = 0
            self.i_update += 1

        return total_err
