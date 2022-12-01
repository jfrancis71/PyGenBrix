import random
import q_model
import cliff_environment

class CliffModel:
    def __init__(self):
        self.env = cliff_environment.CliffEnvironment()
        self.height, self.width = self.env.height, self.env.width

    def sample(self, observation, action):
        self.env.current_state = observation
        return self.env.step(action)


class CliffModelAgent:
    def __init__(self, height, width):
        self.action = None
        self.observation = None
        self.height, self.width = height, width
        self.q_algorithm = q_model.QAlgorithm(CliffModel(), cliff_environment.CliffEnvironment())

    def act(self, observation):
        if random.random() > .1:
            q = self.q_algorithm.q[observation[0], observation[1]]
            action = q.argmax()
        else:
            action = random.sample(range(4), 1)[0]
        self.action = action
        self.observation = observation
        return action

    def observe(self, observation, reward, done, reset):
        self.q_algorithm.update(planning_steps=5)
