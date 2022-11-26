import random
import numpy as np
import copy

# Sutton and Barto Example, page 132
class CliffEnvironment:
    """CliffEnvironment is implementation of Sutton and Barto example on page 132 (2nd Edition)"""
    def __init__(self):
        self.height, self.width = 4, 12
        self.cliff_world = np.zeros([self.height, self.width])
        self.current_state = [self.height-1, 0]
        self.left = 0
        self.up = 1
        self.right = 2
        self.down = 3

    def reset(self):
        """
        Reset the environment

        Sets internal current_state back to the origin.
        """
        self.current_state = [self.height-1, 0]
        return self.current_state

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
        return self.current_state, reward, done, info


class EnvSimulator:
    def __init__(self, state):
        self.env = CliffEnvironment()
        self.env.current_state = state

    def step(self, action):
        return self.env.step(action)

class RandomAgent:
    def act(self, observation):
        return random.sample(range(4), 1)[0]


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
        self.q[observation[0], observation[1], self.action] += .1*\
            ((.99*target_q + reward) - self.q[observation[0], observation[1], self.action])


class LearningSimulator:
    def __init__(self):
        self.state_transition_prob = np.zeros([num_states, num_states])
        self.rewards = np.zeros([num_states, 4])

    def observe(self, observation, action, reward, new_observation, done):
        self.rewards[observation, action] += reward


class QSweepAgent:
    def __init__(self, height, width):
        self.q = np.zeros([height, width, 4])
        self.action = None
        self.observation = None
        self.height, self.width = height, width

    def act(self, observation):
        if random.random() > .1:
            q = self.q[observation[0], observation[1]]
            action = q.argmax()
        else:
            action = random.sample(range(4), 1)[0]
        self.action = action
        self.observation = observation
        return action

    def observe(self, observation, reward, done, reset):
        target_q = self.q[observation[0], observation[1]].max()
        self.q[observation[0], observation[1], self.action] += .1*\
            ((.99*target_q + reward) - self.q[observation[0], observation[1], self.action])
        self.sweep()

    def sweep(self):
        # Iterate through every q state
        for r in range(self.height):
            for c in range(self.width):
                for a in range(4):
                    simulator = EnvSimulator([r, c])
                    new_state, reward, done, info = simulator.step(a)
                    target_q = self.q[new_state[0], new_state[1]].max()
                    if done:
                        target_q = 0.0
                    diff = .1 * \
                        ((.99*target_q + reward) - self.q[r, c, a])
                    self.q[r, c, a] += diff


env = CliffEnvironment()
env.reset()
agent = QSweepAgent(env.height, env.width)

def run_episode():
    done = False
    total_reward = 0
    episode_length = 0
    path = []
    observation = env.reset()
    while not done:
        path.append(observation.copy())
        action = agent.act(observation)
        new_observation, reward, done, info = env.step(action)
        total_reward += reward
        episode_length += 1
        agent.observe(new_observation, reward, done, False)
        observation = new_observation
    path.append(observation.copy())
    print("Path=", path)
    return total_reward, episode_length

for _ in range(25):
    total_reward, episode_length = run_episode()
    print("Total reward=", total_reward, ", Length=", episode_length)