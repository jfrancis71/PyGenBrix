import numpy as np

class BlockingMazeEnvironment:
    """CliffEnvironment is implementation of Sutton and Barto example on page 166 (2nd Edition)"""
    def __init__(self):
        self.height, self.width = 6, 9
        self.cliff_world = np.zeros([self.height, self.width])
        self.current_state = [self.height-1, 3]
        self.left = 0
        self.up = 1
        self.right = 2
        self.down = 3

    def reset(self):
        """
        Reset the environment

        Sets internal current_state back to the origin.
        """
        self.current_state = [self.height-1, 3]
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
            if self.current_state[0] > 0 and ( self.current_state[0] != 4 or self.current_state[1] == 8 ):
                self.current_state[0] -= 1
        elif action == self.down:
            if self.current_state[0] < self.height-1 and ( self.current_state[0] != 2 or self.current_state[1] == 8 ):
                self.current_state[0] += 1
        else:
            print("Unknown action", action)
            raise("Unknown action", action)  # TODO hello
        if self.current_state[0] == 0 and self.current_state[1] == self.width-1:
            done = True
        return self.current_state.copy(), reward, done, info

    def state_to_integer(self, state):
        return state[0]*self.width + state[1]

    def integer_to_state(self, state_no):
        return np.array([state_no/self.width, state_no % self.width], dtype=np.int64)
