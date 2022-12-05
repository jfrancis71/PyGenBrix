import copy
import numpy as np


class QAlgorithm:
    def __init__(self, model, env):
        self.q = np.zeros([model.height, model.width, 4])
        self.model = model
        self.env = env

    def reset(self):
        self.q = self.q * 0.0

    def update(self, planning_steps=1):
        # Iterate through every q state
        for i in range(planning_steps):
            for r in range(self.model.height):
                for c in range(self.model.width):
                    for a in range(4):
                        new_state, reward, done, info = self.model.sample([r, c], a)
                        target_q = self.q[new_state[0], new_state[1]].max()
                        if done:
                            target_q = reward
                        diff = ((.99*target_q + reward) - self.q[r, c, a])
                        self.q[r, c, a] += .1 * diff

    def print(self):
        print("Q:")
        print("  Up")
        print("  ", self.q[:, :, self.env.up])
        print("  Down")
        print("  ", self.q[:, :, self.env.down])
        print("  Right")
        print("  ", self.q[:, :, self.env.right])
        print("  Left")
        print("  ", self.q[:, :, self.env.left])

    def plan(self, state):
        the_plan = []
        for i in range(20):
            action = self.q[state[0], state[1]].argmax()
            new_state, reward, done, info = self.model.sample(state, action)
            the_plan.append([state, action, reward])
            state = copy.deepcopy(new_state)
        return the_plan
