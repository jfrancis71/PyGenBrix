import argparse
import random
import copy
import time
import math
import numpy as np
from pettingzoo.classic import connect_four_v3


def get_legal_actions(env):
    observation, reward, termination, truncation, info = env.last()
    action_mask = observation["action_mask"]
    return np.nonzero(action_mask)[0]


def select_random_action(env):
    legal_actions = get_legal_actions(env)
    num_actions = len(legal_actions)
    return legal_actions[random.randint(0, num_actions - 1)]


class MCTSNode:
    def __init__(self, parent, env):
        self.env = env
        self.sum_value = 0
        self.counts = [0, 0, 0]  # counts of loss, draw, win
        self.count = 0
        self.children = {}  # is a leaf node by default
        self.legal_actions = get_legal_actions(env)
        self.parent = parent
        observation, reward, termination, truncation, info = self.env.last()
        self.terminated = termination
        self.winner = reward
        self.max_depth = 0


def random_rollout(node):
    """does not alter env returns termination reward from perspective of current player"""
    random_env = copy.deepcopy(node.env)
    observation, reward, termination, truncation, info = random_env.last()
    assert termination is False
    swap_reward = -1
    while termination is False:
        random_action = select_random_action(random_env)
        random_env.step(random_action)
        observation, reward, termination, truncation, info = random_env.last()
        swap_reward *= -1
    return reward*swap_reward


def select_node(start_node, tree_selection_policy):
    # ucb rank the children
    if start_node.count == 0:
        return start_node

    selected_node = tree_selection_policy(start_node)
    if selected_node is None:
        return start_node
    else:
        return select_node(selected_node, tree_selection_policy)


def tree_selection_policy_ucb1(node):
    log_parent_n = math.log(node.count)
    best_ucb = -100
    c = math.sqrt(2.0)
    selected_node = None
    for child in node.children.values():
        if child.count == 0:
            ucb = 1000.0
        else:
            ucb = (child.sum_value/child.count) + c * math.sqrt(log_parent_n / child.count)
        if ucb > best_ucb:
            best_ucb = ucb
            selected_node = child
    return selected_node


def tree_selection_policy_ts(node):
    best_sample_value = -1000.0
    selected_node = None
    for child in node.children.values():
        cat_dist = np.random.dirichlet([1+child.counts[0], 1+child.counts[1], 1+child.counts[2]])
        sample = cat_dist[0]*-1 + cat_dist[1]*0 + cat_dist[2]*1
        if sample > best_sample_value:
            best_sample_value = sample
            selected_node = child
    return selected_node


def expand_node(node):
    assert node.children == {}  # we only expand leaf nodes
    for legal_action in node.legal_actions:
        child_env = copy.deepcopy(node.env)
        child_env.step(legal_action)
        node.children[legal_action] = MCTSNode(node, child_env)
    return list(node.children.values())[0]  # returns first of these children


def backup_node(node):
    node = node.parent
    if node is None:
        return
    node.sum_value = 0
    node.counts = [0, 0, 0]
    node.count = 0
    max_depth = -1
    for child in node.children.values():
        node.sum_value -= child.sum_value
        node.count += child.count
        node.counts[0] += child.counts[2]
        node.counts[2] += child.counts[0]
        node.counts[1] += child.counts[1]
        if node.max_depth > max_depth:
            max_depth = child.max_depth
    node.max_depth = max_depth + 1
    backup_node(node)


def mcts(env, num_iterations, tree_selection_policy):
    root_node = MCTSNode(None, env)

    for iterations in range(num_iterations):
        expanding_node = select_node(root_node, tree_selection_policy)
        if expanding_node.terminated == 0:
            rollout_node = expand_node(expanding_node)
            if rollout_node.terminated:
                reward = rollout_node.winner
            else:
                reward = random_rollout(rollout_node)
        else:
            reward = expanding_node.winner
        expanding_node.sum_value += reward
        if reward == 1:
            expanding_node.counts[2] += 1
        elif reward == -1:
            expanding_node.counts[0] += 1
        elif reward == 0:
            expanding_node.counts[1] += 1
        else:
            print("Error")
            quit()
        expanding_node.count += 1
        backup_node(expanding_node)

    max_count = -1
    best_move = None
    max_value = -100
    max_depth = -1
    for p in root_node.children.items():
        if p[1].count > max_count:
            max_count = p[1].count
            if p[1].count > 0:
                max_value = p[1].sum_value / p[1].count
            best_move = p[0]
        if p[1].max_depth > max_depth:
            max_depth = p[1].max_depth

    print("Max Depth=", max_depth)
    for p in root_node.children.items():
        print("Node ", p[0], "sum=", p[1].sum_value, " count = ", p[1].count, " counts=", p[1].counts)
    return max_value, best_move


def board_eval_move(env, depth, cached_evaluations):
    """does not alter env, returns best_score, best_move from perspective of current player"""
    random_action = select_random_action(env)
    if depth == 0:
        return 0, random_action
    legal_actions = get_legal_actions(env)
    scores = []
    for move in legal_actions:
        eval_env = copy.deepcopy(env)
        eval_env.step(move)
        game_state = (tuple(eval_env.env.board), eval_env.agent_selection)
        if game_state in cached_evaluations:
            score = cached_evaluations[game_state]
        else:
            observation, reward, termination, truncation, info = eval_env.last()
            if termination:
                score = reward
            else:
                score = -.99*board_eval_move(eval_env, depth-1, cached_evaluations)[0]
            cached_evaluations[game_state] = score
        scores.append(score)
    best_score = max(scores)
    indices_best_legal_scores = np.nonzero(np.array(scores) == best_score)[0]
    index_best_legal_scores = random.choice(indices_best_legal_scores)
    best_move = legal_actions[index_best_legal_scores]

    return best_score, best_move


def print_board(env):
    board = np.reshape(env.env.board, [6, 7])
    for level in board:
        print(*level, sep=' ')
    print("")


class DualEnvironment:
    """Maintains two separate but almost duplicate environments, one is render, the other is copyable.
    This will only work for deterministic environments.
    Exists because you can't copy.deepcopy environments with PyGame graphical components"""
    def __init__(self, env_fn):
        self.env = env_fn(render_mode="human")
        self.sim_env = env_fn()

    def reset(self):
        self.env.reset()
        self.sim_env.reset()

    def step(self, action):
        self.env.step(action)
        self.sim_env.step(action)

    def last(self):
        observation, reward, termination, truncation, info = self.env.last()
        return observation, reward, termination, truncation, info

    def copy(self):
        return self.sim_env


class MinimaxAgent:
    def __init__(self, depth):
        self.depth = depth

    def act(self, sim_env):
        predicted_score, move = board_eval_move(sim_env, self.depth, {})
        return predicted_score, move


class MCTSAgent:
    def __init__(self, num_iterations, agent_tree_policy):
        self.num_iterations = num_iterations
        self.agent_tree_policy = agent_tree_policy

    def act(self, sim_env):
        predicted_score, move = mcts(sim_env, self.num_iterations, self.agent_tree_policy)
        return predicted_score, move


class HumanAgent:
    def act(self, sim_env):
        move = int(input("Please enter column>"))-1
        return 0.0, move


def play_game(agent1, agent2):
    dual_env.reset()
    winner = 0
    termination = False
    while termination is False:
        print_board(dual_env.env)
        print("Agent1 Move\n")
        start = time.time()
        sim_env = dual_env.copy()
        agent1_predicted_score, agent1_move = agent1.act(sim_env)
        print("Predicting ", agent1_predicted_score, "time=", time.time()-start)
        dual_env.step(agent1_move)
        observation, reward, termination, truncation, info = dual_env.last()
        if termination:
            if reward == 1:
                winner = 1
            else:
                winner = 0
            break
        print("")
        print_board(dual_env.env)
        print("Agent2 Move\n")
        start = time.time()
        sim_env = dual_env.copy()
        agent2_predicted_score, agent2_move = agent2.act(sim_env)
        print("Predicting ", agent2_predicted_score, "time=", time.time()-start)
        dual_env.step(agent2_move)
        observation, reward, termination, truncation, info = dual_env.last()
        if termination:
            if reward == 1:
                winner = 2
            else:
                winner = 0
    print_board(dual_env.env)
    time.sleep(10)
    return winner


def play_tournament(agent1, agent2):
    games = []
    for i in range(10):
        winner = play_game(agent1, agent2)
        print("Winner is ", winner)
        games.append(winner)
        print("Games=", games)
    print("Tournament ", games)


dual_env = DualEnvironment(connect_four_v3.env)

ap = argparse.ArgumentParser(description="Connect 4")
ap.add_argument("--agent1", default="human")
ap.add_argument("--agent2", default="minimax")
ap.add_argument("--agent1_minimax_depth", default=3, type=int)
ap.add_argument("--agent2_minimax_depth", default=3, type=int)
ap.add_argument("--agent1_mcts_num_iterations", default=500, type=int)
ap.add_argument("--agent2_mcts_num_iterations", default=500, type=int)
ap.add_argument("--agent1_tree_policy")
ap.add_argument("--agent2_tree_policy")
ns = ap.parse_args()

if ns.agent1 == "human":
    agent1 = HumanAgent()
elif ns.agent1 == "minimax":
    agent1 = MinimaxAgent(ns.agent1_minimax_depth)
elif ns.agent1 == "mcts":
    if ns.agent1_tree_policy == "ucb1":
        agent1_tree_policy = tree_selection_policy_ucb1
    elif ns.agent1_tree_policy == "ts":
        agent1_tree_policy = tree_selection_policy_ts
    else:
        print("Unknown tree policy for agent1")
        quit()
    agent1 = MCTSAgent(ns.agent1_mcts_num_iterations, agent1_tree_policy)
else:
    print("Unknown agent1")
    quit()

if ns.agent2 == "human":
    agent2 = HumanAgent()
elif ns.agent2 == "minimax":
    agent2 = MinimaxAgent(ns.agent2_minimax_depth)
elif ns.agent2 == "mcts":
    if ns.agent2_tree_policy == "ucb1":
        agent2_tree_policy = tree_selection_policy_ucb1
    elif ns.agent2_tree_policy == "ts":
        agent2_tree_policy = tree_selection_policy_ts
    else:
        print("Unknown tree policy for agent2")
        quit()
    agent2 = MCTSAgent(ns.agent2_mcts_num_iterations, agent2_tree_policy)
else:
    print("Unknown agent2")
    quit()

play_tournament(agent1, agent2)
