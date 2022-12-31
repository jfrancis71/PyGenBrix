import argparse
import numpy as np
import py_cliff_environment
import blocking_maze_environment
import q_agents
import model
import learnable_model
import gym


ap = argparse.ArgumentParser(description="Finite MDP Trainer")
ap.add_argument("--agent")
ap.add_argument("--env", default="PyCliffWalking")
ns = ap.parse_args()


def run_episode():
    done = False
    total_reward = 0
    episode_length = 0
    path = []
    observation = env.reset()
    while not done:
        path.append(observation)
        action = agent.act(observation)
        new_observation, reward, done, info = env.step(action)
        total_reward += reward
        episode_length += 1
        agent.observe(new_observation, reward, done, False)
        observation = new_observation
    path.append(observation)
    print("Path=", path)
    return total_reward, episode_length


class GridToIntegerEnvironment:
    """GridToIntegerEnvironment wraps a grid environment; so state (row,column) pairs are mapped
       to integer state representations.
    """
    def __init__(self, grid_env):
        self.grid_env = grid_env
        self.observation_space = gym.spaces.Discrete(self.grid_env.height*self.grid_env.width)
        self.action_space = gym.spaces.Discrete(self.grid_env.num_actions)

    def reset(self):
        grid_state = self.grid_env.reset()
        state = grid_state[0]*self.grid_env.width + grid_state[1]
        return state

    def step(self, action):
        grid_state, reward, done, info = self.grid_env.step(action)
        state = grid_state[0]*self.grid_env.width + grid_state[1]
        return state, reward, done, info


np.set_printoptions(edgeitems=30, linewidth=100000,
                    formatter=dict(float=lambda x: "%.3g" % x))
if ns.env == "PyCliffWalking":
    env = GridToIntegerEnvironment(py_cliff_environment.GridCliffEnvironment())
elif ns.env == "BlockingMaze":
    env = GridToIntegerEnvironment(blocking_maze_environment.GridBlockingMazeEnvironment())
# Note, CliffWalking slightly different to PyCliffWalking, as walking off cliff takes you back
# to start, therefore cliff states (like end state) are never visited.
elif ns.env == "CliffWalking":
    env = gym.make("CliffWalking-v0", render_mode="human")
elif ns.env == "FrozenLake":
    # our algorithms currently assume done transitions are deterministic, hence set slippery=False
    env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False, max_episode_steps=1000000)
elif ns.env == "Taxi":
    env = gym.make("Taxi-v3", render_mode="human", max_episode_steps=1000000)
else:
    print("Unknown Environment ", ns.env)
    quit()
env.reset()

if ns.agent == "Random":
    agent = q_agents.RandomAgent()
elif ns.agent == "QOnline":
    agent = q_agents.QOnlineAgent(env.height, env.width)
elif ns.agent == "Model":
    agent = model.ModelAgent(env)
elif ns.agent == "LearnableModel":
    agent = learnable_model.DeterministicLearnableModelAgent(env)
elif ns.agent == "StochasticLearnableModel":
    agent = learnable_model.StochasticLearnableModelAgent(env)
else:
    print("Unknown Agent ", ns.agent)
    quit(1)

while True:
    total_reward, episode_length = run_episode()
    print("Total reward=", total_reward, ", Length=", episode_length)
