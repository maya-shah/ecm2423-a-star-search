from copy import copy


def manhattan(start, goal) -> int:
    # returns the heuristic value of Manhattan distance
    start_index = copy(goal)

    for x, y in enumerate(start):
        start_index[y] = x

    goal_index = copy(start)
    for x, y in enumerate(goal):
        goal_index[y] = x

    m = abs(start_index // 3 - goal_index // 3) + abs(start_index % 3 - goal_index % 3)
    return sum(m[1:])
