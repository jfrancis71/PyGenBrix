import random
import copy
import numpy as np
from pettingzoo.classic import connect_four_v3

env = connect_four_v3.env()

env.reset()

def computer_move():
    observation, reward, termination, truncation, info = env.last()
    action_mask = observation["action_mask"]
    num_actions = sum(action_mask)
    return np.nonzero(action_mask)[0][random.randint(0, num_actions - 1)]

def board_eval_human_move(env, depth):
    best_eval = -100
    best_move = -1
    observation, reward, termination, truncation, info = env.last()
    action_mask = observation["action_mask"]
    for move in np.nonzero(action_mask)[0]:
        eval_env = copy.deepcopy(env)
        eval_env.step(move)
        observation, reward, termination, truncation, info = eval_env.last()
        if termination:
            eval = reward
        elif depth>0:
            eval = -board_eval_computer_move(env, depth-1)[1]
        else:
            eval = 0.0
        if eval > best_eval:
            best_eval = eval
            best_move = move
    return best_eval, best_move

def board_eval_computer_move(env, depth):
    best_eval = -100
    best_move = -1
    observation, reward, termination, truncation, info = env.last()
    action_mask = observation["action_mask"]
    for move in np.nonzero(action_mask)[0]:
        eval_env = copy.deepcopy(env)
        eval_env.step(move)
        observation, reward, termination, truncation, info = eval_env.last()
        if termination:
            eval = reward
        elif depth>0:
            eval = -board_eval_human_move(eval_env, depth-1)[1]
        else:
            eval = 0.0
        if eval > best_eval:
            best_eval = eval
            best_move = move
    return best_eval, best_move

def print_board():
    board = np.reshape(env.env.board, [6,7])
    for level in board:
        print(*level, sep=' ')


while True:
    print_board()
    human_move = int(input("Please enter move>"))-1
    env.step(human_move)
    observation, reward, termination, truncation, info = env.last()
    if termination:
        print("Done! ", reward)
        print_board()
        print("Human is victorious")
        quit()
    computers_move = board_eval_computer_move(env, 4)[1]
    env.step(computers_move)
    observation, reward, termination, truncation, info = env.last()
    if termination:
        print("Done! ", reward)
        print_board()
        print("Computer is victorious")
        quit()
