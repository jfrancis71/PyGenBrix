import random
import numpy as np

class RandomAgent:
    def act(self, observation):
        return random.sample(range(4), 1)[0]

    def observe(self, new_observation, reward, done, reset):
        pass


class QOnlineAgent:
    def __init__(self, height, width):
        self.q = np.zeros([height, width, 4])
        self.action = None
        self.observation = None

    def act(self, observation):
        if random.random() > .9:
            q = self.q[observation[0], observation[1]]
            action = q.argmax()
        else:
            action = random.sample(range(4), 1)[0]
        self.action = action
        self.observation = observation
        return action

    def observe(self, observation, reward, done, reset):
        target_q = self.q[observation[0], observation[1]].max()
        self.q[observation[0], observation[1], self.action] += .1 * \
            ((.99*target_q + reward) - self.q[observation[0], observation[1], self.action])
