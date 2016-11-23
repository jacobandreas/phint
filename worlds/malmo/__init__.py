import client_manager
import env_generator
import mission_generator

class MalmoTask(object):
    def __init__(self, mission, env, spec):
        self.mission = mission
        self.env = env
        self.spec = spec
        self.hint = mission.hint

class MalmoWorld(object):
    def __init__(self, config):
        self.config = config
        self.client = client_manager.Client()
        self.n_actions = client_manager.N_ACTIONS
        self.n_features = client_manager.N_FEATURES

    def sample_task(self):
        mission = mission_generator.sample()
        env = env_generator.sample(mission)
        spec = client_manager.create_spec(env, mission)
        return MalmoTask(mission, env, spec)

    def reset(self, tasks):
        assert len(tasks) == 1
        task = tasks[0]
        self.client.reset(task.spec)
        task.mission.prepare(task.env, self.client)
        return self.client.features

    def step(self, actions, tasks):
        assert len(actions) == len(tasks) == 1
        action = actions[0]
        task = tasks[0]
        self.client.step(action)
        reward, stop = task.mission.mark(task.env, self.client)
        return self.client.features, reward, stop
