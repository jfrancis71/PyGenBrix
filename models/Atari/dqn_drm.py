import argparse
import ruamel.yaml as yaml
import pathlib
import tools
import torch
import dreamer as dr
from stable_baselines3 import DQN
from gym import spaces
import numpy as np


class DreamSim():
    def __init__(self, conf, args):
        self.dreamer = dr.Dreamer(config=conf, logger=None, dataset=None)
        _ = self.dreamer.cuda()
        self.dreamer.load_state_dict(torch.load(args.model))
        self.metadata = None
        self.observation_space = spaces.Dict({"stoch": spaces.Box(0,1,[32,32,32]), "deter": spaces.Box(0,1,[600])})
        self.action_space = spaces.Discrete(6)
        env = dr.make_env(conf, None, "evalnocallback", None, None)
        obs = env.reset()
        obs["image"] = np.stack([obs["image"]])
        obs["reward"] = np.array([0.0])
        _, self.init_agent_state = self.dreamer._policy(obs, None, training=False)
        self.init_agent_state = self.init_agent_state[0]
        print("INIT STATE=", self.init_agent_state)
        self.episode = 0
        self.reset()

    def step(self, action):
        action_tensor = torch.tensor([0,0,0,0,0,0]).cuda().unsqueeze(0)*1.0
        action_tensor[0,action] = 1.0
        agent_state = self.agent_state
        agent_state = self.dreamer._task_behavior._world_model.dynamics.img_step(agent_state, action_tensor, sample=self.dreamer._task_behavior._config.imag_sample)
        reward = self.dreamer._wm.heads['reward'](self.dreamer._wm.dynamics.get_feat(agent_state)).mode()[0,0].detach()
        done = self.dreamer._wm.heads['discount'](self.dreamer._wm.dynamics.get_feat(agent_state)).mode()[0,0].detach()
        print("Done=", done)
        self.agent_state = agent_state
        obs = {"stoch": self.agent_state["stoch"].cpu().detach(), "deter": self.agent_state["deter"].cpu().detach()}
        return obs, reward.cpu().numpy(), done, {"episode": self.episode}

    def reset(self):
        self.agent_state = self.init_agent_state
        obs = {"stoch": self.agent_state["stoch"].cpu(), "deter": self.agent_state["deter"].cpu()}
        self.episode += 1
        return obs

#    def observation_space(self):
#        return None

    def reward_range(self):
        return (0,1)

parser = argparse.ArgumentParser()
parser.add_argument('--configs', required=True)
parser.add_argument('--model', required=True)
parser.add_argument("--policy")
parser.add_argument("--random_rollout", action="store_true")
args, remaining = parser.parse_known_args()
configs = yaml.safe_load(
    (pathlib.Path(args.configs)).read_text())
defaults = {}
for name in ["defaults","atari"]:
  defaults.update(configs[name])
parser = argparse.ArgumentParser()
for key, value in sorted(defaults.items(), key=lambda x: x[0]):
  arg_type = tools.args_type(value)
  parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
conf = parser.parse_args([])
conf.traindir = pathlib.Path("~").expanduser()
conf.act = getattr(torch.nn, "ELU")
conf.num_actions = 6

dreamsim = DreamSim(conf, args)

model = DQN('MultiInputPolicy', dreamsim, verbose=1, buffer_size=10000, learning_rate=.0001, learning_starts=1000, target_update_interval=1000)
model.learn(total_timesteps=100000)
