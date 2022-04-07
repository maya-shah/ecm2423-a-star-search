import time
from collections import defaultdict
from copy import deepcopy

import numpy as np

from manhattan import manhattan
from misplaced_tiles import mis_tiles

final_moves = []
final_m = []


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
        heuristic = manhattan(start, goal)
    elif choose_h == '2':
        heuristic = mis_tiles(start, goal)
    return heuristic


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
            fn = gn + hn
            new_state = np.array([(len(state) - 1, fn)],
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
    moves = np.array([("u", [0, 1, 2], -3), ("d", [6, 7, 8], 3), ("r", [2, 5, 8], 1), ("l", [0, 3, 6], -1)],
                     dtype=[('move', str, 1), ('position', list), ('head', int)])

    parent = -1
    gn = 0
    hn = heuristic(start, goal)
    closed = defaultdict(bool)

    d_state = [('start', list), ('parent', int), ('gn', int), ('hn', int)]
    state = np.array([(start, parent, gn, hn)], dtype=d_state)

    d_priority = [('position', int), ('fn', int)]
    priority = np.array([(0, hn)], dtype=d_priority)

    return goal, moves, closed, priority, d_priority, state, d_state


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
        final_moves.append(int(np.where(state[last]['start'] == 0)[0]))
        last = int(state[last]["parent"])
    return optimal.reshape(-1, 3, 3)


def main():
    state, explored = steps(start, goal)
    optimal = generate(state)
    print(("{}\n\nStates Generated: {}\nExplored States: {}"
           "\nThe number of steps needed for the lowest cost: {}").format(
        optimal, len(state), len(state) - explored, len(optimal) - 1))
    for y in range(0, len(final_moves), 1):
        final = final_moves[y] - final_moves[y - 1]
        final_m.append(final)
    for x in range(0, len(final_m)):
        if final_m[x] == -3:
            final_m[x] = 'u'
        elif final_m[x] == 3:
            final_m[x] = 'd'
        elif final_m[x] == 1:
            final_m[x] = 'r'
        elif final_m[x] == -1:
            final_m[x] = 'l'

    print("Moves to final solution: {}".format(final_m[0:-1]))


def solvable(start) -> bool:
    """

    Parameters
    ----------
    start : Array
        The starting array inputted by the user

    Returns
    -------
    bool
        True or False according to if the state given is solvable

    """
    inv_count = 0
    empty_value = -1
    for i in range(0, 9):
        for j in range(i + 1, 9):
            if start[j] != empty_value and start[i] != empty_value and start[i] > start[j]:
                inv_count += 1
    if inv_count % 2 == 0:
        print("\nThis state is solvable.")
        return True
    else:
        print("\nThis state is unsolvable.")
        return False


if __name__ == '__main__':
    startTime = time.time()

    print("This is the general solution for solving any 8-puzzle configuration using A* search methods.")

    start = np.array(
        list(map(int, input("\nPlease input the start state (e.g. 7, 2, 4, 5, 0, 6, 8, 3, 1): ").strip().split(','))))
    print(start.reshape(3, 3))

    goal = np.array(
        list(map(int, input("\nPlease input the goal state (e.g. 1, 2, 3, 4, 5, 6, 7, 0, 8): ").strip().split(','))))
    print(goal.reshape(3, 3))

    if solvable(start):
        pass
    else:
        exit()

    print("\nInput '1' for h1 or '2' for h2.")
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
