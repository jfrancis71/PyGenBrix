import random
import numpy as np
import cliff_environment
import q_model

class QLearnableDeterministicCliffModel:
    def __init__(self):
        self.env = cliff_environment.CliffEnvironment()
        self.height, self.width = self.env.height, self.env.width
        self.dones = np.zeros([self.height, self.width, 4]) - 1
        self.rewards = np.zeros([self.height, self.width, 4])
        num_states = self.height*self.width
        self.state_transitions = np.zeros([num_states, 4, num_states])
        self.visits = np.zeros([self.height, self.width, 4], dtype=np.int64)
        self.random_rewards = np.random.random([self.height, self.width, 4])+50
        self.random_dones = np.random.binomial(1, p=np.ones([self.height, self.width, 4]) - .5)
        self.random_transitions = np.random.randint(num_states, size=[num_states, 4])

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


class CliffLearnableModelAgent:
    def __init__(self, height, width):
        self.action = None
        self.observation = None
        self.height, self.width = height, width
        self.learnable_model = QLearnableDeterministicCliffModel()
        self.q_algorithm = q_model.QAlgorithm(self.learnable_model, self.learnable_model.env)
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
            st = self.learnable_model.env.state_to_integer(observation)
            print("Total explored=", self.learnable_model.visits.sum())
