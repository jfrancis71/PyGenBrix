import random

class RandomAgent:
    def act(self, observation, on_policy):
        return random.sample(range(6), 1)[0]

    def observe(self, new_observation, reward, done, reset):
        pass
