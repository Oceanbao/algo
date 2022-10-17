# If know highest floor of drop without breaking egg (d)
# and a certain # eggs (e). f(d, e)
# Also know the same result for the same # drops and one less egg = f(d, e - 1)
# Now take additional drop from floor f(d, e - 1) + 1
# If breaks, then can the result with one less egg f(d, e - 1)
# Else, can explore the f(d, e) floor above.
# So can get result for a building of 1 + f(d, e - 1) + f(d, e)

# Time O(K log N) since floors[K] grows exponentially with each drop
# Space O(K)

import sys


def superEggDrop(K, N):
    drops = 0
    # floors[i] the # floors that can be checked with i eggs
    floors = [0 for _ in range(K + 1)]

    while floors[K] < N:
        for eggs in range(K, 0, -1):
            floors[eggs] += 1 + floors[eggs - 1]
        drops += 1

    return drops


if __name__ == "__main__":
    EGGS = int(sys.argv[1])
    FLOORS = int(sys.argv[2])

    result = superEggDrop(EGGS, FLOORS)
    print(result)
