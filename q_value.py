import tkinter as tk
import time
import numpy as np
import copy
from random import *
from matinv import *
#from config import initial_grid
import pickle


#grid_value = copy.deepcopy(initial_grid)

with open('all_states_LO3.pkl', 'rb') as f:
    q_values = pickle.load(f)
# Q leaning:
# Environment: ALl possible grid positions, a total of 2^9 = 512


# Actions: clicking one of the 9 grids numbered 0 to 8
# Rewards:

# create a set of all states that are one move away from completing(total of 9 states):
# Each action can be represented as a tuple (i,j) which indicated i-th row, j-th column grid is the button to be clicked.
def states_with_actions(n):
    states_dict = {}
    for i in range(2 ** n):
        binary_array = []
        for j in range(n):
            binary_array.append((i >> j) & 1)
        for x in range(3):
            for y in range(3):
                states_dict[(tuple([tuple(binary_array[i:i + 3]) for i in range(0, 9, 3)]), (x, y))] = 0
    return states_dict


# Reward function:
# --------------------
# After every move, this function compares the current grid state and the initial grid state to compare how many
# lights they have in common using bitwise XOR. The negative of sum of this array will be the reward for that state.
# Note that when the initial grid and the current grid state are the same, the reward is zero (Maximum reward).
# -----------------------

def special_states():
    arrays = []
    for row in range(3):
        for col in range(3):
            empty_grid = [[0, 0, 0] for _ in range(3)]
            empty_grid[row][col] ^= 1
            if row - 1 >= 0:
                empty_grid[row - 1][col] ^= 1  # Cell above
            if row + 1 <= 2:
                empty_grid[row + 1][col] ^= 1  # Cell below
            if col - 1 >= 0:
                empty_grid[row][col - 1] ^= 1  # Cell to the left
            if col + 1 <= 2:
                empty_grid[row][col + 1] ^= 1  # Cell to the right
            arrays.append(empty_grid)
    return arrays


spl_grids = special_states()


def reward_for_state(list_of_lists):
    total = sum([sum(item) for item in list_of_lists])
    if total == 0:
        return 100
    elif list_of_lists in spl_grids:
        return -1
    elif (total > 3):
        return (-1) * total
    else:
        return -5


# Helper functions for the model

# Function to find whether a given state is terminal
def is_terminal(list_of_lists):
    if reward_for_state(list_of_lists) == 100:
        return True
    else:
        return False


# Function to find the max q-value of a state over all actions

def best_action(tuple1, my_dict):
    max_value = float('-inf')  # Initialize with negative infinity
    max_key = None

    for key, value in my_dict.items():
        if key[0] == tuple1 and value > max_value:
            max_value = value
            max_key = key[1]

    return max_key, max_value


def start_action():
    return (np.random.randint(3), np.random.randint(3))


# Epsilon greedy algorithm for choosing the next action
def next_action(grid, epsilon):
    if np.random.random() < epsilon:
        key, value = best_action(grid, q_values)
        return key
    else:
        return (np.random.randint(3), np.random.randint(3))


# Function imitating the action of clicking a grid to toggle the states of itself and its neighbors
def action_click(action, list_of_lists):
    row = action[0]
    col = action[1]
    list_of_lists1 = copy.deepcopy(list_of_lists)
    list_of_lists1[row][col] ^= 1
    if row - 1 >= 0:
        list_of_lists1[row - 1][col] ^= 1  # Cell above
    if row + 1 <= 2:
        list_of_lists1[row + 1][col] ^= 1  # Cell below
    if col - 1 >= 0:
        list_of_lists1[row][col - 1] ^= 1  # Cell to the left
    if col + 1 <= 2:
        list_of_lists1[row][col + 1] ^= 1  # Cell to the right
    return list_of_lists1


def solution(list_of_lists):
    if is_terminal(list_of_lists):
        return []
    else:
        state = (np.random.randint(3), np.random.randint(3))
        solution = []
        solution.append(state)
        list_of_lists1 = action_click(state, list_of_lists)
        while not is_terminal(list_of_lists1):
            grid = tuple([tuple(i) for i in list_of_lists1])
            next_act, val = best_action(grid, q_values)
            list_of_lists1 = action_click(next_act, list_of_lists1)
            if next_act in solution:
                solution.remove(next_act)
            else:
                solution.append(next_act)
        return solution


if __name__ == '__main__':
    q_values = states_with_actions(9)

    epsilon = 0.9
    discount = 0.9
    learning_rate = 0.9

    start = time.time()
    for episode in range(10000):
        initial_grid = []
        for _ in range(3):
            initial_grid.append([randint(0, 1) for b in range(0, 3)])
        grid_value = copy.deepcopy(initial_grid)
        # print(grid_value)
        state = start_action()
        action_click(state)
        while not is_terminal(grid_value):
            # print(grid_value)
            temp_grid = tuple([tuple(i) for i in grid_value])
            next_act = next_action(temp_grid, epsilon)  # next action to take based off of epsilon greedy algorithm
            action_click(next_act)
            reward = reward_for_state(grid_value)
            old_q_value = q_values[(temp_grid, next_act)]
            tuple1 = tuple([tuple(i) for i in grid_value])
            key, max_q_value = best_action(tuple1, q_values)
            temporal_diff = reward + (discount * max_q_value) - old_q_value
            new_q_value = old_q_value + (learning_rate * temporal_diff)
            q_values[(temp_grid, next_act)] = new_q_value

    print('Done')
    end = time.time()
    print(end - start)