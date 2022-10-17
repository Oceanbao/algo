import sys


def super_egg_drop(K, N):
    dp = [[0 for i in range(N + 1)] for _ in range(K + 1)]

    m = 0

    while dp[K][m] < N:
        m += 1
        for k in range(1, K + 1):
            dp[k][m] = dp[k][m - 1] + dp[k - 1][m - 1] + 1

    return m


if __name__ == "__main__":
    EGGS = int(sys.argv[1])
    FLOORS = int(sys.argv[2])

    result = super_egg_drop(EGGS, FLOORS)
    print(result)
