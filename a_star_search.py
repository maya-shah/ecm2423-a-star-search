import time
from collections import defaultdict
from copy import deepcopy

import numpy as np

from manhattan import manhattan
from misplaced_tiles import mis_tiles


def heuristic(start, goal):
    """

    Parameters
    ----------
    start : Array
        Start state
    goal : Array
        Goal state

    Returns
    -------
        heuristic
            The chosen heuristic, Manhattan or Misplaced tiles

    """
    if choose_h == '1':
        chosen_h = manhattan(start, goal)
    elif choose_h == '2':
        chosen_h = mis_tiles(start, goal)
    elif choose_h != '2' or choose_h != '1':
        print("\nNot valid input, please try again and choose '1' or '2'.")
        exit()
    return chosen_h


def steps(start, goal):
    """

    Parameters
    ----------
    start : Array
        Start state
    goal : Array
        Goal state

    Returns
    -------
        state
            states for each move to achieve goal state
        len(priority) : int
            Length of the priority queue
    """
    (goal, moves, closed, priority, d_priority, state, d_state) = setup(start, goal)

    while True:
        priority = np.sort(priority, kind="mergesort",
                           order=['fn', 'position'])
        position = priority[0][0]
        priority = np.delete(priority, 0, 0)
        puzzle = state[position][0]
        gn = state[position][2] + 1
        location = int(np.where(puzzle == 0)[0])

        for move in moves:
            if location not in move['position']:
                new = deepcopy(puzzle)
                head_loc = location + move['head']
                new[location], new[head_loc] = new[head_loc], new[location]
            if closed[tuple(new)]:
                continue
            closed[tuple(new)] = True

            hn = heuristic(new, goal)

            new_state = np.array(
                [(new, position, gn, hn)],
                dtype=d_state)
            state = np.append(state, new_state, 0)

            f_function = gn + hn
            new_state = np.array([(len(state) - 1, f_function)],
                                 dtype=d_priority)
            priority = np.append(priority, new_state, 0)
            if np.array_equal(new, goal):
                print("\nSteps to achieve goal state:\n")
                return state, len(priority)
            else:
                continue


def setup(start, goal):
    """

    Parameters
    ----------
    start : Array
        Start state
    goal : Array
        Goal state

    Returns
    -------
    goal : Array
        The goal array
    moves : Array
        A numpy array containing the possible moves ('up', 'down', 'left', 'right),
            their position, and the head int which moves the puzzle piece in the correct direction
    closed : default-dict
        A default-dict which contain boolean values when checking if a state has already been visited or not
    priority: Array
        A priority array which contains the heuristic values for the states visited
    d_priority : data-type
        The data-type priority array which contains the data-types specified for the priority array,
            'position' and 'fn' (path cost), which are both integers
    state : Array
        The state array which stores the start array, the parent array, gn (parent cost), and the heuristic value
    d_state : data-type
        The data-type state array which contains the data-types specified for the state array,
            'start' (list), 'parent' (int), 'gn' (int), 'hn' (int)

    """
    moves = np.array([("up", [0, 1, 2], -3), ("down", [6, 7, 8], 3), ("left", [2, 5, 8], 1), ("right", [0, 3, 6], -1)],
                     dtype=[('move', str, 1), ('position', list), ('head', int)])

    d_state = [('start', list), ('parent', int), ('gn', int), ('hn', int)]

    parent = -1
    gn = 0
    hn = heuristic(start, goal)
    closed = defaultdict(bool)

    state = np.array([(start, parent, gn, hn)], dtype=d_state)

    d_priority = [('position', int), ('fn', int)]
    priority = np.array([(0, hn)], dtype=d_priority)

    return (goal, moves, closed, priority, d_priority,
            state, d_state)


def generate(state):
    """

    Parameters
    ----------
    state : list
        A list containing all the possible states

    Returns
    -------
    optimal : list
        A list containing all the possible states in matrix form

    """
    optimal = np.array([], int)
    last = len(state) - 1
    while last != -1:
        optimal = np.insert(optimal, 0, state[last]['start'])
        last = int(state[last]["parent"])
    return optimal.reshape(-1, 3, 3)


def main():
    state, explored = steps(start, goal)
    optimal = generate(state)
    print(("{}\n\nStates Generated: {}\nExplored States: {}"
           "\nThe number of steps needed for the lowest cost: {}").format(
        optimal, len(state), len(state) - explored, len(optimal) - 1))


if __name__ == "__main__":
    startTime = time.time()

    start = np.array([7, 2, 4,
                      5, 0, 6,
                      8, 3, 1])

    goal = np.array([1, 2, 3,
                     4, 5, 6,
                     7, 0, 8])

    print("Input '1' for h1 or '2' for h2.")
    choose_h = input("A* (h1): Manhattan\nA* (h2): Misplaced Tiles\n")

    heuristic(start, goal)

    print("Start 8 Puzzle:")
    print(start.reshape(3, 3))

    print("\nGoal 8 Puzzle:")
    print(goal.reshape(3, 3))

    main()

    if heuristic(start, goal) == manhattan(start, goal):
        print("Manhattan Distance: ", manhattan(start, goal))
    elif heuristic(start, goal) == mis_tiles(start, goal):
        print("Misplaced Tiles: ", mis_tiles(start, goal))

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
