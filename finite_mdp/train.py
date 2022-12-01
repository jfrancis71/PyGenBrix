import argparse
import numpy as np
import cliff_environment
import q_agents
import cliff_model
import cliff_learnable_model


ap = argparse.ArgumentParser(description="Finite MDP Trainer")
ap.add_argument("--agent")
ns = ap.parse_args()


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
env = cliff_environment.CliffEnvironment()
env.reset()
if ns.agent == "Random":
    agent = q_agents.RandomAgent()
elif ns.agent == "QOnline":
    agent = q_agents.QOnlineAgent(4, 12)
elif ns.agent == "CliffModel":
    agent = cliff_model.CliffModelAgent(4, 12)
elif ns.agent == "CliffLearnableModel":
    agent = cliff_learnable_model.CliffLearnableModelAgent(4, 12)

else:
    print("Unknown Agent ", ns.agent)
    quit(1)

for _ in range(25):
    total_reward, episode_length = run_episode()
    print("Total reward=", total_reward, ", Length=", episode_length)
