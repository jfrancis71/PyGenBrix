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

    def state_to_integer(self, state):
        return state[0]*self.width + state[1]

    def integer_to_state(self, state_no):
        return np.array([state_no/self.width, state_no % self.width], dtype=np.int64)


class CliffModel:
    def __init__(self):
        self.env = CliffEnvironment()
        self.height, self.width = self.env.height, self.env.width

    def sample(self, observation, action):
        self.env.current_state = observation
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
        self.q[observation[0], observation[1], self.action] += .1 * \
            ((.99*target_q + reward) - self.q[observation[0], observation[1], self.action])


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
                            target_q = 0.0
                        diff = .1 * \
                            ((.99*target_q + reward) - self.q[r, c, a])
#                        if r == 2 and c == 11 and a == self.env.down:
#                            print("target_q for 2,11,down", target_q, " and done=", done)
                        self.q[r, c, a] += diff

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
            new_state = state
        return the_plan


class CliffModelAgent:
    def __init__(self, height, width):
        self.action = None
        self.observation = None
        self.height, self.width = height, width
        self.q_algorithm = QAlgorithm(CliffModel())

    def act(self, observation):
        if random.random() > .1:
            q = self.q_algorithm.q[observation[0], observation[1]]
            action = q.argmax()
        else:
            action = random.sample(range(4), 1)[0]
        self.action = action
        self.observation = observation
        return action

    def observe(self, observation, reward, done, reset):
        self.q_algorithm.update(planning_steps=5)


class LearnableDeterministicCliffModel:
    def __init__(self):
        self.env = CliffEnvironment()
        self.height, self.width = self.env.height, self.env.width
        self.dones = np.zeros([self.height, self.width, 4]) - 1
        self.rewards = np.zeros([self.height, self.width, 4])
        num_states = self.height*self.width
        self.state_transitions = np.zeros([num_states, 4, num_states])
        self.visits = np.zeros([self.height, self.width, 4], dtype=np.int64)
        self.random_rewards = np.random.random([self.height, self.width, 4])+50
        self.random_dones = np.random.binomial(1, p=np.ones([self.height, self.width, 4]) - .5)
        self.random_transitions = np.random.randint(num_states, size=[num_states, 4])
        print("DEBUG", self.random_transitions.shape)

    def sample(self, observation, action):
        self.env.current_state = observation.copy()
        done = self.dones[observation[0], observation[1], action]
        reward = self.rewards[observation[0], observation[1], action]
        observation_state = self.env.state_to_integer(observation)
        next_state = self.state_transitions[observation_state, action].argmax()
        actual_new_observation, actual_reward, actual_done, info = self.env.step(action)
        if self.visits[observation[0], observation[1], action] == 0:
            reward = self.random_rewards[observation[0], observation[1], action]
            done = self.random_dones[observation[0], observation[1], action]
            next_state = self.random_transitions[observation_state, action]
        new_observation = self.env.integer_to_state(next_state)
        return new_observation, reward, done, info

    def observe(self, observation, action, reward, done, next_observation):
        self.dones[observation[0], observation[1], action] = done
        self.visits[observation[0], observation[1], action] = 1
        self.rewards[observation[0], observation[1], action] = reward
        observation_state = self.env.state_to_integer(observation)
        next_observation_state = self.env.state_to_integer(next_observation)
        self.state_transitions[observation_state, action, next_observation_state] = 1.0

    def sample_parameters(self):
        self.random_dones = np.random.binomial(1, p=np.ones([self.height, self.width, 4]) - .5)
        self.random_rewards = np.random.random([self.height, self.width, 4])+50
        num_states = self.height * self.width
        self.random_transitions = np.random.randint(num_states, size=[num_states, 4])

    def print_dones(self):
        print("Dones:")
        print("  Up")
        print("  ", self.dones[:, :, self.env.up])
        print("  Down")
        print("  ", self.dones[:, :, self.env.down])
        print("  Right")
        print("  ", self.dones[:, :, self.env.right])
        print("  Left")
        print("  ", self.dones[:, :, self.env.left])

    def print_visits(self):
        print("Visits:")
        print("  Up")
        print("  ", self.visits[:, :, self.env.up])
        print("  Down")
        print("  ", self.visits[:, :, self.env.down])
        print("  Right")
        print("  ", self.visits[:, :, self.env.right])
        print("  Left")
        print("  ", self.visits[:, :, self.env.left])

    def print_sampled_rewards(self):
        sampled_rewards = self.visits*self.rewards + (1-self.visits)*self.random_rewards
        print("Sampled Rewards:")
        print("  Up")
        print("  ", sampled_rewards[:, :, self.env.up])
        print("  Down")
        print("  ", sampled_rewards[:, :, self.env.down])
        print("  Right")
        print("  ", sampled_rewards[:, :, self.env.right])
        print("  Left")
        print("  ", sampled_rewards[:, :, self.env.left])

    def print_sampled_transitions(self):
        get_one_hot = self.state_transitions.reshape([4, 12, 4, 48]).argmax(axis=3)
        sampled_transitions = self.visits * get_one_hot
        print("Sampled Transitions:")
        print("  Up")
        print("  ", sampled_transitions[:, :, self.env.up])
        print("  Down")
        print("  ", sampled_transitions[:, :, self.env.down])
        print("  Right")
        print("  ", sampled_transitions[:, :, self.env.right])
        print("  Left")
        print("  ", sampled_transitions[:, :, self.env.left])


class CliffLearnableAgent:
    def __init__(self, height, width):
        self.action = None
        self.observation = None
        self.height, self.width = height, width
        self.learnable_model = LearnableDeterministicCliffModel()
        self.q_algorithm = QAlgorithm(self.learnable_model, self.learnable_model.env)
        self.steps = 0

    def act(self, observation):
        if random.random() > .0:
            q = self.q_algorithm.q[observation[0], observation[1]]
            action = q.argmax()
        else:
            action = random.sample(range(4), 1)[0]
        self.action = action
        self.observation = observation
        self.steps += 1
        return action

    def observe(self, observation, reward, done, reset):
        self.learnable_model.observe(self.observation, self.action, reward, done, observation)
        if self.steps % 20 == 0:
            self.learnable_model.sample_parameters()
            self.q_algorithm.reset()
            self.q_algorithm.update(planning_steps=40)
            self.q_algorithm.update(planning_steps=40)
            self.q_algorithm.update(planning_steps=40)
            plan = self.q_algorithm.plan(observation)
            print("The Plan: ", plan)
            ac = plan[0][1]
#            print("Visit count for 1st action", self.learnable_model.visits[observation[0], observation[1], ac])
            st = self.learnable_model.env.state_to_integer(observation)
#            print("From ", st, " to Next state: ", self.learnable_model.state_transitions[st,ac])
            print("Total explored=", self.learnable_model.visits.sum())
#            print("1st Q Action for ", observation, "=", self.q_algorithm.q[observation[0], observation[1] ])


def run_episode():
    done = False
    total_reward = 0
    episode_length = 0
    path = []
    observation = env.reset()
    while not done:
        path.append(observation)
        assert observation == env.current_state
        action = agent.act(observation)
        new_observation, reward, done, info = env.step(action)
        total_reward += reward
        episode_length += 1
        agent.observe(new_observation, reward, done, False)
        observation = new_observation
    path.append(observation)
    print("Path=", path)
    return total_reward, episode_length


np.set_printoptions(edgeitems=30, linewidth=100000,
                    formatter=dict(float=lambda x: "%.3g" % x))
env = CliffEnvironment()
env.reset()
agent = CliffLearnableAgent(env.height, env.width)

for _ in range(25):
    total_reward, episode_length = run_episode()
    print("Total reward=", total_reward, ", Length=", episode_length)
