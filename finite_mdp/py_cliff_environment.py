import numpy as np
import gym

# Sutton and Barto Example, page 132
class GridCliffEnvironment:
    """GridCliffEnvironment is implementation of Sutton and Barto example on page 132 (2nd Edition)"""
    def __init__(self):
        self.height, self.width = 4, 12
        self.cliff_world = np.zeros([self.height, self.width])
        self.current_state = [self.height-1, 0]
        self.left = 0
        self.up = 1
        self.right = 2
        self.down = 3
        self.num_actions = 4

    def reset(self):
        """
        Reset the environment

        Sets internal current_state back to the origin.
        """
        self.current_state = [self.height-1, 0]
        return self.current_state.copy()

    def step(self, action):
        info = None
        done = False
        reward = -1.0
        if action == self.left:
            if self.current_state[1] > 0:
                self.current_state[1] -= 1
        elif action == self.right:
            if self.current_state[1] < self.width-1:
                self.current_state[1] += 1
        elif action == self.up:
            if self.current_state[0] > 0:
                self.current_state[0] -= 1
        elif action == self.down:
            if self.current_state[0] < self.height-1:
                self.current_state[0] += 1
        else:
            print("Unknown action", action)
            raise("Unknown action", action)  # TODO hello
        if self.current_state[0] == self.height-1 and 1 <= self.current_state[1] <= self.width-2:
            reward -= 100.0
        if self.current_state[0] == self.height-1 and self.current_state[1] == self.width-1:
            done = True
        return self.current_state.copy(), reward, done, info
