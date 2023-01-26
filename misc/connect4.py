import random
import copy
import time
import math
import numpy as np
from pettingzoo.classic import connect_four_v3


def get_legal_actions(env):
    observation, reward, termination, truncation, info = env.last()
    action_mask = observation["action_mask"]
    return action_mask

def select_random_action(env):
    action_mask = get_legal_actions(env)
    num_actions = sum(action_mask)
    return np.nonzero(action_mask)[0][random.randint(0, num_actions - 1)]


class MCTSNode:
    def __init__(self, parent, env):
        self.env = env
        self.sum_value = 0
        self.count = 0
        self.children = {}  # is a leaf node by default
        self.legal_actions = get_legal_actions(env)
        self.parent = parent
        observation, reward, termination, truncation, info = self.env.last()
        self.terminated = termination
        self.winner = reward

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


def select_node(start_node):
    # ucb rank the children
    if start_node.count == 0:
        return start_node
    log_parent_n = math.log(start_node.count)
    best_ucb = -100
    c = math.sqrt(2.0)
    selected_node = None
    for child in start_node.children.values():
        if child.count == 0:
            ucb = 1000.0
        else: # note change sign, their interest opposite to mine
            ucb = (child.sum_value/child.count) + c * math.sqrt(log_parent_n / child.count)
        if ucb > best_ucb:
            best_ucb = ucb
            selected_node = child
    if selected_node is None:
        return start_node
    else:
        return select_node(selected_node)

def expand_node(node):
    assert node.children == {}  # we only expand leaf nodes
    for legal_action in np.nonzero(node.legal_actions)[0]:
        child_env = copy.deepcopy(node.env)
        child_env.step(legal_action)
        node.children[legal_action] = MCTSNode(node, child_env)
    return list(node.children.values())[0]  # returns first of these children

def backup_node(node):
    node = node.parent
    if node is None:
        return
    node.sum_value = 0
    node.count = 0
    for child in node.children.values():
        node.sum_value -= child.sum_value
        node.count += child.count
    backup_node(node)

def mcts(env):
    root_node = MCTSNode(None, env)

    for iterations in range(500):
        expanding_node = select_node(root_node)
        if expanding_node.terminated == 0:
            rollout_node = expand_node(expanding_node)
            if rollout_node.terminated:
                reward = rollout_node.winner
            else:
                reward = random_rollout(rollout_node)
        else:
            reward = expanding_node.winner
        expanding_node.sum_value += reward
        expanding_node.count += 1
        backup_node(expanding_node)

    max_count = -1
    best_move = None
    max_value = -100
    for p in root_node.children.items():
        if p[1].count > max_count:
            max_count = p[1].count
            max_value = p[1].sum_value / p[1].count
            best_move = p[0]
    for p in root_node.children.items():
        print("Node ", p[0], "sum=", p[1].sum_value, " count = ", p[1].count)
    return max_value, best_move


def board_eval_move(env, depth, cached_evaluations):
    """does not alter env, returns best_score, best_move from perspective of current player"""
    random_action = select_random_action(env)
    if depth == 0:
        return 0, random_action
    action_mask = get_legal_actions(env)
    legal_moves = np.nonzero(action_mask)[0]
    scores = []
    for move in legal_moves:
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
    best_move = legal_moves[index_best_legal_scores]
    if action_mask[best_move] != 1:
        print("Internal Error")
#    assert action_mask[best_move] == 1
    if depth == 6:
        print("Returning final answer")
        print("best_move=", best_move, " legal_moves=", legal_moves, " scores = ", scores, " best_score=", best_score, "indices=", indices_best_legal_scores, " idx=", index_best_legal_scores)

    return best_score, best_move

def print_board(env):
    board = np.reshape(env.env.board, [6,7])
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


def play_game():
    dual_env.reset()
    winner = 0
    while winner == 0:
        print_board(dual_env.env)
    #    human_move = int(input("Please enter move>"))-1
        print("Human Move\n")
        sim_env = dual_env.copy()
        human_predicted_score, human_move = board_eval_move(sim_env, 6, {})
        print("Predicting ", human_predicted_score)
        dual_env.step(human_move)
        observation, reward, termination, truncation, info = dual_env.last()
        if termination:
            winner = 1
            break
        print("")
        print_board(dual_env.env)
        print("Computer Move\n")
#        computers_predicted_score, computers_move = board_eval_move(eval_env, 6, {})
        sim_env = dual_env.copy()
        computers_predicted_score, computers_move = mcts(sim_env)
        print("Predicting ", computers_predicted_score)
        dual_env.step(computers_move)
        observation, reward, termination, truncation, info = dual_env.last()
        if termination:
            winner = 2
    print_board(dual_env.env)
    time.sleep(10)
    return winner

dual_env = DualEnvironment(connect_four_v3.env)

def tournament():
    games = []
    for i in range(10):
        winner = play_game()
        print("Winner is ", winner)
        games.append(winner)
        print("Games=", games)
    print("Tournament ", games)

tournament()
