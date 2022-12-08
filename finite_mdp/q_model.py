import copy
import numpy as np


class QAlgorithm:
    def __init__(self, mdp):
        self.q = np.zeros([mdp.num_states, 4])
        self.mdp = mdp

    def reset(self):
        self.q = self.q * 0.0

    def update(self, planning_steps=1):
        # Iterate through every q state
        for i in range(planning_steps):
            for s in range(self.mdp.num_states):
                for a in range(4):
                    new_state, reward, done, info = self.mdp.sample(s, a)
                    target_q = self.q[new_state].max()
                    if done:
                        target_q = reward
                    diff = ((.99*target_q + reward) - self.q[s, a])
                    self.q[s, a] += .1 * diff

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
            action = self.q[state].argmax()
            new_state, reward, done, info = self.mdp.sample(state, action)
            the_plan.append([state, action, reward])
            state = copy.deepcopy(new_state)
        return the_plan