import random
import copy
import q_model

class Model:
    def __init__(self, env):
        self.env = copy.deepcopy(env)
        self.height, self.width = self.env.height, self.env.width

    def sample(self, observation, action):
        self.env.current_state = observation
        return self.env.step(action)


class ModelAgent:
    def __init__(self, env):
        self.action = None
        self.observation = None
        self.q_algorithm = q_model.QAlgorithm(Model(env), env)
        self.num_actions = env.action_space()

    def act(self, observation):
        if random.random() > .1:
            q = self.q_algorithm.q[observation[0], observation[1]]
            action = q.argmax()
        else:
            action = random.sample(range(self.num_actions), 1)[0]
        self.action = action
        self.observation = observation
        return action

    def observe(self, observation, reward, done, reset):
        self.q_algorithm.update(planning_steps=5)
