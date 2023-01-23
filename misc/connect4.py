import random
import copy
import time
import numpy as np
from pettingzoo.classic import connect_four_v3

env = connect_four_v3.env(render_mode="human")
eval_env = connect_four_v3.env()

env.reset()

def get_legal_actions(env):
    observation, reward, termination, truncation, info = env.last()
    action_mask = observation["action_mask"]
    return action_mask

def select_random_action(env):
    action_mask = get_legal_actions(env)
    num_actions = sum(action_mask)
    return np.nonzero(action_mask)[0][random.randint(0, num_actions - 1)]

def board_eval_move(env, depth, cached_evaluations):
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
#        print("Game state=", game_state)
        if game_state in cached_evaluations:
            score = cached_evaluations[game_state]
#            print("Hitting cache", game_state, ", score=", score)
        else:
            observation, reward, termination, truncation, info = eval_env.last()
            if termination:
                score = reward
            else:
                score = -.99*board_eval_move(eval_env, depth-1, cached_evaluations)[0]
            cached_evaluations[game_state] = score
        scores.append(score)
    best_score = max(scores)
    idxs_best_legal_scores = np.nonzero(np.array(scores) == best_score)[0]
    idx_best_legal_scores = random.choice(idxs_best_legal_scores)
    best_move = legal_moves[idx_best_legal_scores]
    if action_mask[best_move] != 1:
        print("Internal Error")
#    assert action_mask[best_move] == 1
    if depth == 6:
        print("Returning final answer")
        print("best_move=", best_move, " legal_moves=", legal_moves, " scores = ", scores, " best_score=", best_score, "idxs=", idxs_best_legal_scores, " idx=", idx_best_legal_scores)

    return best_score, best_move

def print_board():
    board = np.reshape(env.env.board, [6,7])
    for level in board:
        print(*level, sep=' ')
    print("")

def play_game():
    env.reset()
    eval_env.reset()
    winner = 0
    while winner == 0:
        print_board()
    #    human_move = int(input("Please enter move>"))-1
        print("Human Move\n")
        human_pred_score, human_move = board_eval_move(eval_env, 3, {})
        print("Predicting ", human_pred_score)
        env.step(human_move)
        eval_env.step(human_move)
        observation, reward, termination, truncation, info = env.last()
        if termination:
            winner = 1
            break
        print("")
        print_board()
        print("Computer Move\n")
        computers_pred_score, computers_move = board_eval_move(eval_env, 6, {})
        print("Predicting ", computers_pred_score)
        env.step(computers_move)
        eval_env.step(computers_move)
        observation, reward, termination, truncation, info = env.last()
        if termination:
            winner = 2
    print_board()
    time.sleep(10)
    return winner

games = []
for i in range(10):
    winner = play_game()
    print("Winner is ", winner)
    games.append(winner)
    print("Games=", games)
print("Tournament ", games)
