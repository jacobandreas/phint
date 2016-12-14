from misc import util
import net

import bisect
from collections import defaultdict, namedtuple
import logging
import numpy as np
import tensorflow as tf

#N_BATCH = 2000
N_BATCH = 1000
N_HISTORY = 20000
N_STEPS_PER_UPDATE = 5

N_HINT = 5

N_RECURRENT = 64
N_HIDDEN = 64
N_EMBED = 64

N_TASKS = 100

DISCOUNT = 0.9

InputBundle = namedtuple("InputBundle", ["t_hint", "t_init_obs", "t_obs",
    "t_action_mask", "t_reward", "t_oldprob"])

StateRepresenter = namedtuple("StateRepresenter", ["t_init_repr", "t_repr", "vars"])
TaskRepresenter = namedtuple("TaskRepresenter", ["t_attention", "t_task_repr", "vars"])
Policy = namedtuple("Policy", ["t_action", "t_baseline", "action_vars", "baseline_vars", "t_bias_op"])
Trainer = namedtuple("Trainer", ["t_action_err", "t_train_action_op", "t_baseline_err", "t_train_baseline_op"])

ModelState = namedtuple("ModelState", ["hint", "init_obs", "index"])

class ModularModel(object):
    def __init__(self, config, world):
        self.experiences = []
        self.good_experiences = []
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

        self.next_hint_action = world.n_actions
        self.n_actions = self.next_hint_action + 1

        #self.optimizer = tf.train.RMSPropOptimizer(0.0003)
        self.optimizer = tf.train.RMSPropOptimizer(0.0001)
        #self.optimizer = tf.train.RMSPropOptimizer(0.003)

        def build_input_bundle():
            t_hint = tf.placeholder(tf.int32, shape=(None, N_HINT))
            with tf.variable_scope("init_obs"):
                t_init_obs = tf.placeholder(tf.float32, shape=(None, self.n_features))
            t_obs = tf.placeholder(tf.float32, shape=(None, self.n_features))
            t_action_mask = tf.placeholder(tf.float32, shape=(None, self.n_actions))
            t_reward = tf.placeholder(tf.float32, shape=(None,))
            t_oldprob = tf.placeholder(tf.float32, shape=(None,))
            return InputBundle(t_hint, t_init_obs, t_obs, t_action_mask, t_reward, t_oldprob)

        def build_state_representer(name, inputs):
            t_init_obs = inputs.t_init_obs
            t_obs = inputs.t_obs
            with tf.variable_scope(name) as scope:
                t_init_hidden, _ = net.mlp(t_init_obs, (N_HIDDEN,), final_nonlinearity=True)
                scope.reuse_variables()
                t_hidden, _ = net.mlp(t_obs, (N_HIDDEN,), final_nonlinearity=True)

                vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.name)
                return StateRepresenter(t_init_hidden, t_hidden, vars)

        def build_task_representer(name, inputs, state_repr):
            t_hint = inputs.t_hint
            # TODO naming consistency
            t_init_repr = state_repr.t_init_repr
            with tf.variable_scope(name) as scope:
                t_embed_hint, _ = net.embed(t_hint, N_TASKS, N_EMBED)
                t_attention_key, _ = net.mlp(t_init_repr, (N_EMBED,))
                t_attention_rs = tf.reshape(t_attention_key, (-1, 1, N_HIDDEN))
                t_attention_tile = tf.tile(t_attention_rs, (1, N_HINT, 1))
                t_attention_mul = t_attention_tile * t_embed_hint
                t_attention_unnorm = tf.exp(tf.reduce_sum(t_attention_mul,
                        reduction_indices=(2,), keep_dims=True))
                t_normalizer = tf.reshape(tf.reduce_sum(t_attention_unnorm,
                        reduction_indices=(1, 2)), (-1, 1, 1))
                with tf.variable_scope("attention"):
                    t_attention = t_attention_unnorm / t_normalizer
                t_hint_weighted = t_embed_hint * t_attention
                t_task_repr = tf.reduce_sum(t_hint_weighted, reduction_indices=(1,))

                vars = tf.get_collection(tf.GraphKeys.VARIABLES,
                        scope=scope.name) + state_repr.vars
                return TaskRepresenter(t_attention, t_task_repr, vars)

        def build_policy(name, state_repr, task_repr):
            t_state_repr = state_repr.t_repr
            t_task_repr = task_repr.t_task_repr
            with tf.variable_scope(name) as scope:
                t_hidden, common_vars = net.mlp(tf.concat(1, (t_task_repr, t_state_repr)),
                        (N_HIDDEN,), final_nonlinearity=True)
                #t_hidden, common_vars = net.mlp(t_state_repr, (N_HIDDEN,),
                #        final_nonlinearity=True)
                with tf.variable_scope("action"):
                    t_action_scores, action_vars = net.mlp(t_hidden, (self.n_actions,))
                    #t_action = tf.nn.log_softmax(t_action_scores)
                    t_action = tf.nn.softmax(t_action_scores)
                    t_bias_op = action_vars[1].assign(action_vars[1]-3.)
                with tf.variable_scope("baseline"):
                    #t_baseline, baseline_vars = net.mlp(t_hidden, (1,))
                    t_baseline = tf.get_variable("baseline",
                            shape=(), initializer=tf.constant_initializer(0))
                    baseline_vars = [t_baseline]

                all_common_vars = common_vars + state_repr.vars #+ task_repr.vars
                all_action_vars = set(all_common_vars + action_vars)
                all_baseline_vars = set(all_common_vars + baseline_vars)
                return Policy(t_action, t_baseline, all_action_vars,
                        all_baseline_vars, t_bias_op)

        def build_trainer(inputs, policy):
            t_action_mask = inputs.t_action_mask
            t_reward = inputs.t_reward
            t_action = policy.t_action
            t_baseline = policy.t_baseline

            t_advantage = t_reward - t_baseline

            t_chosen_prob = tf.reduce_sum(t_action_mask * t_action, reduction_indices=(1,))
            t_ratio = tf.stop_gradient(t_chosen_prob / inputs.t_oldprob)
            t_pg = tf.log(t_chosen_prob) * t_advantage
            t_ppo = tf.minimum(t_ratio * t_pg, tf.clip_by_value(t_ratio, 0.8, 1.2) * t_pg)
            #t_ppo = tf.minimum(t_ratio * t_pg, tf.clip_by_value(t_ratio, 0, 1) * t_pg)
            t_action_err = -tf.reduce_mean(t_ppo)
            t_train_action_op = self.optimizer.minimize(t_action_err, var_list=policy.action_vars)
            #self.t_ratio = -(tf.log(t_chosen_prob) * t_advantage)
            #self.t_ratio = tf.log(t_chosen_prob) * t_advantage
            #self.t_ratio = t_action_err

            t_baseline_err = tf.reduce_mean(tf.square(t_advantage))
            t_train_baseline_op = self.optimizer.minimize(t_baseline_err, var_list=policy.baseline_vars)
            #t_baseline_err, t_train_baseline_op = 0, 0

            return Trainer(t_action_err, t_train_action_op, t_baseline_err, t_train_baseline_op)

        self.inputs = build_input_bundle() 
        self.state_repr = build_state_representer("state_repr", self.inputs)
        self.task_repr = build_task_representer("task_repr", self.inputs, self.state_repr)
        #self.task_repr = None
        self.policy = build_policy("policy", self.state_repr, self.task_repr)
        self.trainer = build_trainer(self.inputs, self.policy)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())
        self.session.run([self.policy.t_bias_op])

        self.task_index = util.Index()

        self.summary_writer = tf.train.SummaryWriter(
                self.config.experiment_dir+"/tf", 
                self.session.graph)

        self.i_iter = 0

    def init(self, hints):
        embedded = [tuple(self.task_index.index(subtask) for subtask in hint) for
                hint in hints]
        self.mstates = [ModelState(e, np.zeros((self.n_features,)), 0) for e in embedded]
        return self.mstates

    def experience(self, episode):
        running_reward = 0
        for transition in episode[::-1]:
            running_reward = running_reward * DISCOUNT + transition.r
            n_transition = transition._replace(r=running_reward)
            if n_transition.a < self.n_actions:
                self.experiences.append(n_transition)
                if n_transition.r > 0.1:
                    self.good_experiences.append(n_transition)
        self.experiences = self.experiences[-N_HISTORY:]
        self.good_experiences = self.good_experiences[-N_HISTORY:]
        #self.good_experiences = sorted(self.good_experiences, key=lambda t:
        #        t.r)[-N_HISTORY:]

    def act(self, obs):
        hints = np.zeros((len(obs), N_HINT))
        init_obs = np.zeros((len(obs), self.n_features))
        attention = np.zeros((len(obs), N_HINT, 1))
        for i, m in enumerate(self.mstates):
            hints[i, :len(m.hint)] = m.hint
            init_obs[i, :] = m.init_obs
            if m.index < len(m.hint):
                attention[i, m.index, 0] = 1
        #print attention[:, 0, 0].astype(int)
        feed_dict = {
            self.inputs.t_hint: hints,
            self.inputs.t_init_obs: init_obs,
            self.task_repr.t_attention: attention,
            self.inputs.t_obs: np.asarray(obs),
        }
        probs, = self.session.run([self.policy.t_action], feed_dict=feed_dict)
        out = []
        out_p = []
        stop = []
        mstates = [None] * len(obs)
        for i_state in range(len(obs)):
            mstate = self.mstates[i_state]
            action = self.random.choice(self.n_actions, p=probs[i_state, :])
            #if action == self.attend_hint_action:
            #    self.mstates[i_state] = \
            #        self.mstates[i_state]._replace(init_obs=obs[i])
            if action == self.next_hint_action:
                new_index = mstate.index + 1
                mstate = mstate._replace(index=new_index)
            mstates[i_state] = mstate
            out.append(action)
            out_p.append(probs[i_state, action])
            stop.append(mstate.index == len(mstate.hint))
        self.mstates = mstates
        return out, out_p, mstates, stop

    def train(self):
        if len(self.experiences) < N_HISTORY:
            return None
        #print len(self.experiences)

        total_err = 0
        for i_step in range(N_STEPS_PER_UPDATE):
            #batch = self.experiences
            if len(self.good_experiences) > 0: # and self.i_iter < 1000:
                good_indices = [self.random.randint(len(self.good_experiences)) for
                        _ in range(min(N_BATCH/5, len(self.good_experiences)))]
            else:
                good_indices = []
            all_indices = [self.random.randint(len(self.experiences)) for _ in
                    range(N_BATCH - len(good_indices))]
            all_batch = [self.experiences[i] for i in all_indices]
            good_batch = [self.good_experiences[i] for i in good_indices]
            batch = all_batch + good_batch

            hints = np.zeros((N_BATCH, N_HINT))
            states0 = np.zeros((N_BATCH, self.n_features))
            attention = np.zeros((N_BATCH, N_HINT, 1))
            states1 = np.zeros((N_BATCH, self.n_features))
            states2 = np.zeros((N_BATCH, self.n_features))
            action_masks = np.zeros((N_BATCH, self.n_actions))
            rewards = np.zeros(N_BATCH)
            oldprobs = np.zeros(N_BATCH)
            for i_experience in range(N_BATCH):
                exp = batch[i_experience]
                hints[i_experience, :len(exp.m1.hint)] = exp.m1.hint
                states0[i_experience, :] = exp.m1.init_obs
                attention[i_experience, exp.m1.index, 0] = 1
                states1[i_experience, :] = exp.s1
                states2[i_experience, :] = exp.s2
                action_masks[i_experience, exp.a] = 1
                rewards[i_experience] = exp.r
                oldprobs[i_experience] = exp.p
            feed_dict = {
                self.inputs.t_hint: hints,
                self.inputs.t_init_obs: states0,
                self.task_repr.t_attention: attention,
                self.inputs.t_obs: states1,
                self.inputs.t_action_mask: action_masks,
                self.inputs.t_reward: rewards,
                self.inputs.t_oldprob: oldprobs
            }

            #t_ratio, = self.session.run([self.t_ratio], feed_dict=feed_dict)
            #print t_ratio
            #exit()

            action_err, _ = self.session.run(
                    [self.trainer.t_action_err, self.trainer.t_train_action_op],
                    feed_dict=feed_dict)

            baseline_err, _ = self.session.run(
                    [self.trainer.t_baseline_err, self.trainer.t_train_baseline_op], 
                    feed_dict=feed_dict)
            #baseline_err = 0
            total_err += np.asarray([action_err, baseline_err])

        #self.experiences = []

        self.i_iter += 1

        return total_err
