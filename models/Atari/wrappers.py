import numpy as np
import gym


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

