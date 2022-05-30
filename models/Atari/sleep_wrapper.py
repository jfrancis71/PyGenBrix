import gym
import time

class SleepWrapper(gym.Wrapper):
    def __init__(self, env, sleep=0.02):
        super(SleepWrapper, self).__init__(env)
        self.sleep = sleep

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        time.sleep(self.sleep)
        return obs, reward, done, info
