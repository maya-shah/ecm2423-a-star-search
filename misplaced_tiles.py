def mis_tiles(x, y) -> int:
    # variable h = 0, represents the heuristic value
    h = 0
    for n, m in zip(x, y):
        if n != 0 and n != m:
            h += 1
    return h
