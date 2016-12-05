from misc import util
import net

from collections import defaultdict, namedtuple
import logging
import numpy as np
import tensorflow as tf

N_BATCH = 256
N_HISTORY = 10
N_STEPS_PER_UPDATE = 2
#N_EXPERIENCES = 10000
N_EPISODES = 1000
N_UPDATE = 50

N_RECURRENT = 64
N_HIDDEN = 64
N_EMBED = 64

N_TASKS = 100

DISCOUNT = 0.9

InputBundle = namedtuple("InputBundle", ["t_hint", "t_obs", "t_obs_seq",
    "t_obs_seq_target", "t_rnn_state", "t_action_mask", "t_reward"])
QBundle = namedtuple("QBundle", ["t_q", "t_next_rnn_state", "t_q_seq", "t_q_chosen", "t_q_best", "vars"])
ModelState = namedtuple("ModelState", ["hint", "rnn_state"])

class MemoryModel(object):
    def __init__(self, config, world):
        self.episodes = []
        tf.set_random_seed(0)
        self.config = config
        self.random = np.random.RandomState(0)
        self.prepare(world)

    def prepare(self, world):
        self.world = world
        self.n_features = world.n_features
        self.n_actions = world.n_actions

        self.optimizer = tf.train.AdamOptimizer(0.001)

        # shared
        t_hint = tf.placeholder(tf.int32, shape=(None, None))

        # one step
        t_obs = tf.placeholder(tf.float32, shape=(None, self.n_features))
        t_rnn_state = tf.placeholder(tf.float32, shape=(None, N_RECURRENT))

        # history batch
        t_obs_seq = tf.placeholder(tf.float32, shape=(None, N_HISTORY, self.n_features))
        t_obs_seq_target = tf.placeholder(tf.float32, shape=(None, N_HISTORY, self.n_features))
        t_action_mask = tf.placeholder(tf.float32, shape=(None, N_HISTORY, self.n_actions))
        t_reward = tf.placeholder(tf.float32, shape=(None, N_HISTORY))

        def build_net(name, t_hint, t_obs, t_obs_seq, t_action_mask):
            with tf.variable_scope(name) as scope:
                # shared
                t_embed_hint, _ = net.embed(t_hint, N_TASKS, N_EMBED)
                cell = tf.nn.rnn_cell.GRUCell(N_RECURRENT)
                cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, self.n_actions)

                # one step
                t_q = None
                t_next_rnn_state = None
                if t_obs is not None:
                    t_hidden, _ = net.mlp(t_obs, (N_HIDDEN,), final_nonlinearity=True)
                    tt_hidden = [t_hidden]
                    tt_q, tt_next_rnn_state = tf.nn.seq2seq.attention_decoder(
                            tt_hidden, t_rnn_state, t_embed_hint, cell)
                    t_q = tt_q[0]
                    t_next_rnn_state = tt_next_rnn_state[0]
                    scope.reuse_variables()

                # history batch
                t_hidden_seq, _ = net.mlp(t_obs_seq, (N_HIDDEN,), final_nonlinearity=True)
                tt_hidden_seq = tf.unpack(t_hidden_seq, num=N_HISTORY, axis=1)
                tt_q_seq, _ = tf.nn.seq2seq.attention_decoder(
                        tt_hidden_seq, t_rnn_state, t_embed_hint, cell)
                t_q_seq = tf.pack(tt_q_seq, axis=1)
                
                t_q_seq_chosen = None
                if t_action_mask is not None:
                    t_q_seq_chosen = tf.reduce_sum(t_action_mask * t_q_seq, reduction_indices=(2,))
                t_q_seq_best = tf.reduce_max(t_q_seq, reduction_indices=(2,))

                vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.name)

                #zero = cell.zero_state(1, tf.float32)[0, :]
                zero = np.zeros(N_RECURRENT)

                return QBundle(t_q, t_next_rnn_state, t_q_seq, t_q_seq_chosen, t_q_seq_best, vars), zero

        qnet, self.zero_state = build_net("net", t_hint, t_obs, t_obs_seq, t_action_mask)
        qnet_target, _ = build_net("net_target", t_hint, None, t_obs_seq_target, None)

        t_td = t_reward + DISCOUNT * qnet_target.t_q_best - qnet.t_q_chosen
        t_err = tf.reduce_mean(tf.square(t_td))

        t_train_op = self.optimizer.minimize(t_err, var_list=qnet.vars)
        t_update_ops = [v_target.assign(v) for v, v_target in zip(qnet.vars, qnet_target.vars)]

        self.qnet = qnet
        self.qnet_target = qnet_target
        self.t_err = t_err
        self.t_train_op = t_train_op
        self.t_update_ops = t_update_ops
        self.inputs = InputBundle(t_hint, t_obs, t_obs_seq, t_obs_seq_target,
                t_rnn_state, t_action_mask, t_reward)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

        self.i_step = 0
        self.i_task_step = defaultdict(lambda: 0)
        self.i_update = 0

        self.task_index = util.Index()

    def init(self, hints):
        embedded = [tuple(self.task_index.index(subtask) for subtask in hint) for
                hint in hints]
        self.mstates = [ModelState(e, self.zero_state) for e in embedded]
        return self.mstates

    def experience(self, episode):
        print len(episode)
        self.episodes.append(episode)
        self.episodes = self.episodes[-N_EPISODES:]
        self.i_task_step[episode[0].m1.hint] += 1

    def act(self, obs):
        feed_dict = {
            self.inputs.t_obs: np.asarray(obs),
            self.inputs.t_hint: np.asarray([m.hint for m in self.mstates]),
            self.inputs.t_rnn_state: np.asarray([m.rnn_state for m in self.mstates])
        }
        q, nstate = self.session.run([self.qnet.t_q, self.qnet.t_next_rnn_state], feed_dict=feed_dict)
        out = []
        for i_state in range(len(obs)):
            hint = self.mstates[i_state].hint
            if self.random.rand() < max(0.1, (30 - self.i_task_step[hint]) / 30.):
                out.append(self.random.randint(self.n_actions))
            else:
                out.append(np.argmax(q[i_state, :]))
            self.mstates[i_state] = self.mstates[i_state]._replace(rnn_state=nstate.copy())
        return zip(out, self.mstates)

    def train(self):
        logging.info("[train] %d", self.i_step)
        episodes = self.episodes

        #if len(experiences) < N_BATCH * N_STEPS_PER_UPDATE:
        #if len(episodes) < N_BATCH:
        if len(sum(episodes, [])) < N_BATCH:
            return None

        logging.info("[begin loop]")
        total_err = 0
        for i_step in range(N_STEPS_PER_UPDATE):
            self.i_step += 1
            batch_indices = [self.random.randint(len(episodes)) for _ in range(N_BATCH)]
            batch_episodes = [self.episodes[i_exp] for i_exp in batch_indices]
            batch_offsets = [self.random.randint(len(e)) for e in batch_episodes]
            batch = [e[o:o+N_HISTORY] for e, o in zip(batch_episodes, batch_offsets)]
            lens = [len(e) for e in batch]

            states1 = np.zeros((N_BATCH, N_HISTORY, self.n_features))
            states2 = np.zeros((N_BATCH, N_HISTORY, self.n_features))
            action_masks = np.zeros((N_BATCH, N_HISTORY, self.n_actions))
            rewards = np.zeros((N_BATCH, N_HISTORY))
            hints = np.zeros((N_BATCH, max(len(ep[0].m1.hint) for ep in batch)))
            rnn_states = np.zeros((N_BATCH, N_RECURRENT))
            for i_episode in range(N_BATCH):
                for i_step in range(len(batch[i_episode])):
                    exp = batch[i_episode][i_step]
                    states1[i_episode, i_step, :] = exp.s1
                    states2[i_episode, i_step, :] = exp.s2
                    action_masks[i_episode, i_step, exp.a] = 1
                    rewards[i_episode, i_step] = exp.r
                hints[i_episode] = exp.m1.hint
                rnn_states[i_episode] = exp.m1.rnn_state
            feed_dict = {
                self.inputs.t_obs_seq: states1,
                self.inputs.t_hint: hints,
                self.inputs.t_obs_seq_target: states2,
                self.inputs.t_action_mask: action_masks,
                self.inputs.t_reward: rewards,
                self.inputs.t_rnn_state: rnn_states,
            }
            err, _ = self.session.run([self.t_err, self.t_train_op], feed_dict=feed_dict)
            total_err += err

        if self.i_step >= N_UPDATE:
            logging.info("[update] %d", self.i_update)
            self.session.run(self.t_update_ops)
            self.i_step = 0
            self.i_update += 1

        return total_err
