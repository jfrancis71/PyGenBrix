#Uses pytorch implementation of dreamerv2: https://github.com/jsikyoon/dreamer-torch
#Original codebase: https://github.com/danijar/dreamerv2
#Original paper: https://arxiv.org/abs/2010.02193
#@misc{hafner2021mastering,
#      title={Mastering Atari with Discrete World Models}, 
#      author={Danijar Hafner and Timothy Lillicrap and Mohammad Norouzi and Jimmy Ba},
#      year={2021},
#      eprint={2010.02193},
#      archivePrefix={arXiv},
#      primaryClass={cs.LG}
#}
#Modified fork with all_actions = False.
#Atari Pong mcts, policy rollout, 
#AC unrestricted actions, 20
#MCTS use policy value, unresticted, -11


import argparse
import ruamel.yaml as yaml
import tools
import dreamer as dr
import pathlib
import torch.nn
import functools
import numpy as np
import sys
import treelib
import math


class MCTSNode():
    def __init__(self, parent):
        self.w = 0.0
        self.n = 0
        self.parent = parent
        self.parent_action = None
        self.children = {}

    def ancestors(self):
        current = self
        actions = []
        while current.parent is not None:
            actions.append(current.parent_action)
            current = current.parent
        actions.reverse()
        return actions

    def print_ancestors(self):
        current = self
        while current.parent is not None:
            print( current.ancestors(), "w=", current.w.cpu().detach().numpy(), "n=", current.n, "score=", current.score().cpu().detach().numpy())
            current = current.parent

    def score(self):
        return self.w/self.n

    def ucb(self, N):
        return np.random.random()
        return self.score() + .2 * math.sqrt ( math.log(self.parent.n) / self.n )
        
#Useful references:
#http://ccg.doc.gold.ac.uk/ccg_old/papers/browne_tciaig12_1.pdf
#https://www.youtube.com/watch?v=UXW2yZndl7U&t=628s
class MCTS():
    def __init__(self, random_rollout=True):
        self.root = MCTSNode(parent=None)
        self.actions = 6
        self.random_rollout = random_rollout

    def exec_actions(self, actions, agent_state):
        agent_states = [agent_state]
        rewards = []
        for action in actions:
            action_tensor = torch.tensor([0,0,0,0,0,0]).cuda().unsqueeze(0)*1.0
            action_tensor[0,action] = 1.0
            action = action_tensor
            agent_state = dreamer._task_behavior._world_model.dynamics.img_step(agent_state, action, sample=dreamer._task_behavior._config.imag_sample)
            reward = dreamer._wm.heads['reward'](dreamer._wm.dynamics.get_feat(agent_state)).mode()[0,0].detach()
            agent_states.append(agent_state)
            rewards.append(reward)
        return agent_states, rewards

    def get_rollout_from_leaf_node(self, leaf_node, agent_state):
        """returns the total rollout value from a node"""
        actions = leaf_node.ancestors()
        agent_states, rewards = self.exec_actions(actions, agent_state)#?rewards accumulate
        agent_state = agent_states[-1]
        value1 = sum(rewards)
        value2 = self.get_rollout_from_state(agent_state, 15)
        value = value1 + value2
        return value

    def get_rollout_from_state(self, agent_state, horizon):
        if self.random_rollout is False:
            feat = dreamer._wm.dynamics.get_feat(agent_state)
            value = dreamer._task_behavior.value(feat).mode()
            return value
        actions = []
        for _ in range(horizon):
            action = np.random.randint(low=0,high=self.actions)
            actions.append(action)
        _, rewards = self.exec_actions(actions, agent_state)
        return sum(rewards)

    def get_rollout_from_leaf_node_n(self, leaf_node, agent_state, n):
        cum = 0.0
        for _ in range(n):
            cum += self.get_rollout_from_leaf_node(leaf_node, agent_state)
        return cum/n

    def explore(self): #aka selection
        """returns the leaf node with highest UCB"""
        node = self.root
        while node.children != {}:
            unexplored = False
            for child in node.children.values():
                if child.n == 0:
                    node = child
                    unexplored = True
            if unexplored:
                continue
            vlist = [ v.ucb(1) for v in node.children.values()]
            vmax = max(vlist)
            for v in range(self.actions):
                if vlist[v] == vmax:
                    node = node.children[v]
                    continue

        return node

    def backpropogate(self, leaf_node):
        """backpropogates UCB, note nothing to do with NN backpropogation, this is MCTS backpropogation!"""
        node = leaf_node
        while node.parent is not None:
            node = node.parent
            node.w = sum([child_node.w for child_node in node.children.values()])
            node.n = sum([child_node.n for child_node in node.children.values()])

    def expand(self, leaf_node):
        for action in range(self.actions):
            child = MCTSNode(parent=leaf_node)
            child.parent_action = action
            child.parent = leaf_node
            leaf_node.children[action] = child

    def run(self, agent_state, horizon):
        while horizon > 0:
            explore_node = self.explore()
            value = self.get_rollout_from_leaf_node_n(explore_node, agent_state, 1)

            self.expand(explore_node)
            explore_node.w += value
            explore_node.n += 1
            #fill in values?
            self.backpropogate(explore_node)
            horizon -= 1

        current_max = -100.0
        action = torch.tensor([1,0,0,0,0,0]).cuda().unsqueeze(0)*1.0
        current_max = -100.0
        for act in range(self.actions):
            if self.root.children[act].n > 0:
                w = (self.root.children[act].w/self.root.children[act].n).detach().cpu().numpy()
                if w > current_max:
                    current_max = w
                    action = torch.tensor([0,0,0,0,0,0]).cuda().unsqueeze(0)*1.0
                    action[0,act] = 1
        return action


def mcts_policy(agent_state):
    tree = MCTS(random_rollout=False)
    action = tree.run(agent_state, 64)
    return action.detach()[0].cpu().numpy()

def ac_policy(agent_state):
    feat = dreamer._wm.dynamics.get_feat(agent_state)
    action_dist = dreamer._task_behavior.actor(feat)
    action = action_dist.mode()
    return action.detach()[0].cpu().numpy()

def random_policy(agent_state):
    action_dist = tools.OneHotDist(1.0*(torch.tensor([0,0,0,0,0,0])).cuda()[None])
    action = action_dist.sample()
    return action.detach()[0].cpu().numpy()


class Game():
    def __init__(self, policy):
        if policy == "ac":
            self.policy = ac_policy
        elif policy == "random":
            self.policy = random_policy
        elif policy == "mcts":
            self.policy = mcts_policy
        else:
            assert False

    def step(self, agent_state):
        action = self.policy(agent_state[0])
        act = { "action": action }
        env_obs, reward, done, info = eval_env.step(act)
        eval_env.render(mode="human")
        obs["image"] = np.stack([env_obs["image"]])
        _, agent_state = dreamer._policy(obs, agent_state, training=False)
        return agent_state, env_obs, reward, done, info

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

eval_env = dr.make_env(conf, None, "evalnocallback", None, None)

conf.traindir = pathlib.Path("~").expanduser()
conf.act = getattr(torch.nn, "ELU")
acts = eval_env.action_space
conf.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]

dreamer = dr.Dreamer(config=conf, logger=None, dataset=None)
_ = dreamer.cuda()
dreamer.load_state_dict(torch.load(args.model))

obs = eval_env.reset()
obs["image"] = np.stack([obs["image"]])
obs["reward"] = np.array([1.0])

if __name__ == "__main__":
    agent_state = None
    steps = 0
    cum_reward = 0
    done = False
    game = Game(args.policy)
    action, agent_state = dreamer._policy(obs, agent_state, training=False)
    while not done:
        agent_state, env_obs, reward, done, info = game.step(agent_state)
        cum_reward += reward
        steps += 1
    print("Info=", cum_reward)
    print("Step=", steps)
