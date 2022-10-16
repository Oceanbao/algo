import numpy as np


def minDistance(s1, s2):
    m = len(s1)
    n = len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=np.int16)
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + 1,
                )
    print(dp)


minDistance("rap", "apple")
