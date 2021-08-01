#Achieves about -8 after an hour of training

import gym

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import *


class DisplayEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, ac):
        rets = self.env.step(ac)
        self.env.render()
        return rets


#altered from Deep Reinforcement Learning Hands-On, Maxim Lapan, 2nd edition
class StackFramesWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(StackFramesWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=2),
                                                old_space.high.repeat(n_steps, axis=2), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:,:,:-1] = self.buffer[:,:,1:]
        self.buffer[:,:,-1] = observation[:,:,0]
        return self.buffer

env = gym.make("PongNoFrameskip-v4")
env.seed(42)
env = MaxAndSkipEnv(env, skip=4)
env = WarpFrame(env)
env = StackFramesWrapper(env, 2, np.uint8)
model = DQN('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=2500000)
