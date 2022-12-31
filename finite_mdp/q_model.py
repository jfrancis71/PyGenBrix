import copy
import numpy as np
import torch
import torch.nn as nn


class QAlgorithm(nn.Module):
    def __init__(self, mdp):
        super().__init__()
        self.q = nn.Parameter(torch.zeros([mdp.num_states, mdp.num_actions]), requires_grad=False)
        self.mdp = mdp

    def reset(self):
        self.q *=  0.0

    def update(self, planning_steps=1):
        # Iterate through every q state
        for i in range(planning_steps):
             next_states, rewards, dones, info = self.mdp.samples()
             target_q = torch.index_select(self.q,0,next_states.flatten()).reshape([self.mdp.num_states,self.mdp.num_actions,self.mdp.num_actions]).max(dim=2)[0] * (1-dones) + dones*rewards
             diff = (.99*target_q + rewards) - self.q
             self.q += .1 * diff.detach()

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
