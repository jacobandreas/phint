from misc import util
import net

from collections import namedtuple
import logging
import numpy as np
import tensorflow as tf

N_BATCH = 256
N_STEPS_PER_UPDATE = 10
N_EXPERIENCES = 10000
N_UPDATE = 100

N_HIDDEN = 64
N_EMBED = 64

N_TASKS = 100

DISCOUNT = 0.9

InputBundle = namedtuple("InputBundle", ["t_task", "t_obs", "t_obs_target", "t_action_mask", "t_reward"])
QBundle = namedtuple("QBundle", ["t_q", "t_q_chosen", "t_q_best", "vars"])
ModelState = namedtuple("ModelState", ["task"])

class SimpleQModel(object):
    def __init__(self, config, world):
        self.experiences = []
        tf.set_random_seed(0)
        self.config = config
        self.random = np.random.RandomState(0)
        self.prepare(world)

    def prepare(self, world):
        self.world = world
        self.n_features = world.n_features
        self.n_actions = world.n_actions

        self.optimizer = tf.train.AdamOptimizer(0.001)

        t_obs = tf.placeholder(tf.float32, shape=(None, self.n_features))
        t_obs_target = tf.placeholder(tf.float32, shape=(None, self.n_features))
        t_task = tf.placeholder(tf.int32, shape=(None,))
        t_action_mask = tf.placeholder(tf.float32, shape=(None, self.n_actions))
        t_reward = tf.placeholder(tf.float32, shape=(None,))

        def build_net(name, t_task, t_obs, t_action_mask=None):
            with tf.variable_scope(name):
                t_embed_task, v_embed_task = net.embed(t_task, N_TASKS, N_EMBED)
                t_input = tf.concat(1, (t_obs, t_embed_task))
                t_q, v_q = net.mlp(t_input, (N_HIDDEN, self.n_actions))
                if t_action_mask is None:
                    t_q_chosen = None
                else:
                    t_q_chosen = tf.reduce_sum(t_action_mask * t_q, reduction_indices=(1,))
                t_q_best = tf.reduce_max(t_q, reduction_indices=(1,))
                return QBundle(t_q, t_q_chosen, t_q_best, v_embed_task + v_q)

        qnet = build_net("net", t_task, t_obs, t_action_mask)
        qnet_target = build_net("net_target", t_task, t_obs_target)

        t_td = t_reward + DISCOUNT * qnet_target.t_q_best - qnet.t_q_chosen
        t_err = tf.reduce_mean(tf.square(t_td))

        t_train_op = self.optimizer.minimize(t_err, var_list=qnet.vars)
        t_update_ops = [v_target.assign(v) for v, v_target in zip(qnet.vars, qnet_target.vars)]

        self.qnet = qnet
        self.qnet_target = qnet_target
        self.t_err = t_err
        self.t_train_op = t_train_op
        self.t_update_ops = t_update_ops
        self.inputs = InputBundle(t_task, t_obs, t_obs_target, t_action_mask, t_reward)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

        self.i_step = 0
        self.i_update = 0

        self.task_index = util.Index()

    def init(self, tasks):
        self.mstates = [ModelState(self.task_index.index(t)) for t in tasks]
        return self.mstates

    def experience(self, episode):
        self.experiences.extend(episode)
        self.experiences = self.experiences[-N_EXPERIENCES:]

    def act(self, obs):
        feed_dict = {
            self.inputs.t_obs: obs,
            self.inputs.t_task: [m.task for m in self.mstates]
        }
        q = self.session.run([self.qnet.t_q], feed_dict=feed_dict)[0]
        out = []
        for i_state in range(len(obs)):
            if self.random.rand() < max(0.1, (10 - self.i_update) / 10.):
                out.append(self.random.randint(self.n_actions))
            else:
                out.append(np.argmax(q[i_state, :]))
        return zip(out, self.mstates)

    def train(self):
        experiences = self.experiences
        #if len(experiences) < N_BATCH * N_STEPS_PER_UPDATE:
        if len(experiences) < N_BATCH:
            return None

        total_err = 0
        for i_step in range(N_STEPS_PER_UPDATE):
            self.i_step += 1
            batch_indices = [np.random.randint(len(experiences)) for _ in range(N_BATCH)]
            batch = [self.experiences[i_exp] for i_exp in batch_indices]
            s1, m1, a, s2, m2, r = zip(*batch)
            action_mask = np.zeros((len(batch), self.n_actions))
            for i_datum, action in enumerate(a):
                action_mask[i_datum, action] = 1
            feed_dict = {
                self.inputs.t_obs: s1,
                self.inputs.t_task: [mm1.task for mm1 in m1],
                self.inputs.t_obs_target: s2,
                self.inputs.t_action_mask: action_mask,
                self.inputs.t_reward: r
            }
            err, _ = self.session.run([self.t_err, self.t_train_op], feed_dict=feed_dict)
            total_err += err

        logging.info("[train] %d", self.i_step)
        if self.i_step >= N_UPDATE:
            logging.info("[update] %d", self.i_update)
            self.session.run(self.t_update_ops)
            self.i_step = 0
            self.i_update += 1

        return total_err
