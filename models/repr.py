from misc import util
from net import _linear, _embed

class ReprModel(object):
    def __init__(self, config, world, guide):
        self.world = world
        self.guide = guide
        self.config = config
        self.prepare(config, world, guide)
        self.saver = tf.train.Saver()

    def prepare(self, config, world, guide):
        self.t_obs = tf.placeholder(tf.float32, (None, world.n_obs))
        self.controller = EmbeddingController(config, self.t_obs, world, guide)
        self.actor = Actor(config, self.t_obs, self.controller.t_repr, world, guide)
        self.critic = Critic(config, self.t_obs, self.controller.t_repr, world, guide)
        self.t_action_param = self.actor.t_action_param

    def init(self, task, obs):
        return None

    def feed(self, obs):
        return {self.t_obs: obs}

    def act(self, obs, mstate, task, session):
        n_obs = len(obs)
        action_p = session.run([self.t_action_param], self.feed(obs))
        action, ret = self.action_dist.sample(action_p)
        return zip(action, ret), [False]*n_obs, None, [0]*n_obs

    def save(self, session):
        self.saver.save(session, os.path.join(self.config.experiment_dir, "repr.chk"))
