# ALGO

## Framework learning DS and Algo

### Storage mode of DS

- There are only two ways to store DS: array and linked list
- Array and linked-list are structural basis while superstructure can be viewed as special operations with diff APIs
- Queue and stack can be implemented by both, with diff pros and cons
- Graph can be impl either - adjacency table is linked-list, adjacency matrix is 2D array; AM can be used to eval connectivity quickly and can solve problems via matrix operations, but if graph is sparse, this become time-consuming; AT is more space-saving, but the efficiency of many ops is certainly less than for an adjacency matrix
- Hashtables map keys to large array by making use of hash function to solve hash conflicts;- Trees with array is HEAP, as it's a complete binary tree; BST, AVL trees, red-black trees, interval trees, B-trees, etc. based on list
- Arrays: compact and continuous storage with random access; find and search fast via index and space-saving; O(n) in reallocation, insertion is O(n)
- List: insert and deletion O(1); access is costly with more storage need for pointers

### Basic operations on DS

- No more than traversal + access, add, delete, search and modify
- Linear traversal (for / while) and nonlinear is recursion
- Array traversal framework, typical linear iterative for loop
- List traversal framework, both iterative and recursive
- Binary tree traversal framework, nonlinear recursive
- N-tree traversal can be extended to graph traversal as graph is a combo of several n-tree;

### Guidelines of Algo Exercises

- Do binary tree first! Easiest to train framework thinking and most of algo skills are essentially tree traversal problems
- Almost all topics of binary trees are a set of this framework:

```c
void traverse(TreeNode root) {
    // pre-order traversal
    traverse(root.left)
    // middle-order traversal
    traverse(root.right)
    // post-order traversal
}
```

- e.g. Leetcode No.124, hard. Find max sum of paths

```c
int ans = INT_MIN;
int oneSideMax(TreeNode* root) {
    if (root == nullptr) return 0;
    int left = max(0, oneSideMax(root->left));
    int right = max(0, oneSideMax(root->right));
    ans = max(ans, left + right + root->val);
    return max(left, right) + root->val;
}
```

- after doing binary tree traversal and learn the frameworks, then do any backtracking, DP, you will find that as long as recursion is involved, it's all a tree problem

## DP

- The overlapped problems, best sub-structure and state transition equation are the 3 elements
- it is hardest to write out the state transition equation
- **find out state -> define DP array -> find out option -> find out base case**

### Fibonacci sequence

1. brute force recursion

```c
int fib(int N) {
    if (N == 1 || N == 2) return 1;
    return fib(N - 1) + fib(N - 2);
}
```

- **overlapped subproblem**: Why inefficient? To draw out the tree structure: number of nodes in a binary tree is O(2^n) and that is the time complexity

2. recursive solution with memos

```c
int fib(int N) {
    if (N < 1) return 0;
    vector<int> memo(N + 1, 0);
    return helper(memo, N);
}

int helper(vector<int>& memo, int n) {
    if (n == 1 || n == 2) return 1;
    if (memo[n] != 0) return memo[n];
    memo[n] = helper(memo, n - 1) + helper(memo, n - 2);
    return memo[n];
}
```

3. recursive answer to DP array

- DP table (memo) and in it to complete 'bottom up' is not beautiful

```c
int fib(int N) {
    vector<int> dp(N + 1, 0);
    dp[1] = dp[2] = 1;
    for (int i = 3; i <= N; i++)
      dp[i] = dp[i - 1] + dp[i - 2];
    return dp[N];
}
```

- **state-transition equation** is the Fib equation; **state transfer equation** the hardest part, the brute-force solution - the optim is nothing more than the use of DP table
- since only two states (n, n-1) are needed at any time, the space can be reduced to O(1)

```c
int fib(int n) {
    if (n == 2 || n == 1)
      return 1;
    int prev = 1, curr = 1;
    for (int i = 3; i <= n; i++) {
        int sum = prev + curr;
        prev = curr;
        curr = sum;
    }
    return curr;
}
```

### Collecting change

COINS in diff denom of 'k', c1, c2...cK, the num of each coin is unlimited, and then given total, find at least how many COINS needed to scrape up, -1 if impossible.

```c
int coinChange(int[] coins, int amount);
```

E.g. k=3, face value 1,2,5 total = 11; so at least 3 COINS where 11 = 5 + 5 + 1
Idea: enumerate all possible combo and find min COINS

1. brute-force

- DP as it has 'optimal substructure' where each independent of each other;
- Think, to get highest total, sub-problem is to get highest for each
- Transition equation - first 'state' - variable that changes in the original problem and subproblems - COINS is INF, the only state is target amount
- then find DP function: current target n, at least dp(n) are needed
- then find 'choice', for each state what choices can be made to change the current state - no matter what the target amount is, the choice is to choose a coin from denomiation list COINS, then target amount will be reduced:
- finally base case: when target is 0, COINS required is 0; less than 0 no solution, -1

```python
def coinChange(coins: List[int], amount: int):
  def dp(n):
    # base case
    if n == 0: return 0
    if n < 0: return -1
    # to min it is to init to INF
    res = float('INF')
    for coin in coins:
      subproblem = dp(n - coin)
      # no solution to subproblem, skip
      if subproblem == -1: continue
      res = min(res, 1 + subproblem)

    return res if res != float('INF') else -1

  return dp(amount)
```

- state transfer equation: dp(n) = 
- (a) 0, n = 0
- (b) -1, n < 0
- (c) min( dp(n - coin) + 1 | coin in coins), n > 0
- now eliminate overlapping subproblem 

2. Recursion with memo

```python
def coinChange(coins: List[int], amount: int):
  memo = {}
  def dp(n):
    if n in memo: return memo[n]
    # ...same
    memo[n] = res if res != float('INF') else -1
    return memo[n]
  return dp(amount)
```

- num of subproblem is O(n), unit processing time is O(k), so total time complexity is O(kn)

3. Iterative solution of DP array

- DP table from bottom-up to eliminate overlapping subproblems
- **dp[I] = x means that when the target amont is I, at least x COINS needed**

```c
int coinChange(vector<int>& coins, int amount) {
    // array size is amount + 1 (initail value)
    vector<int> dp(amount + 1, amount + 1);
    // base
    dp[0] = 0;
    for (int i = 0; i < dp.size(); i++) {
        // inner loop to find min of + 1 for all subproblem
        for (int coin : coins) {
          if (i - coin < 0) continue;
          dp[i] = min(dp[i], 1 + dp[i - coin]);
        }
    }
    return (dp[amount] == amount + 1) ? -1 : dp[amount];
}
```

Epilogue

- Computer solving problem, simply exhaust all possibilities
- Algo is about how to exhaust, then efficiently

### Classic DP: Edit Distance

Problem: min step to convert word1 to word2 with 3 ops (insert, delete, replace)

Idea: two pointers i,j starting from backward
Base: when i == j and when either i or j is index-0, the other side can only insert the rest of the other

```python
if s1[i] == s2[j]:
  skip
  i, j move forward
else:
  chose:
    insert
    delete
    replace
```

- How to choose the 3? Try all and pick the smallest - recursive

```python
def minDistance(s1, s2) -> int:

  # memo
  memo = {}
  
  # return the least editing distance s1[0..i] and s2[0..j]
  def dp(i, j):
    if (i, j) in memo:
      return memo[(i, j)]

    # base
    if i == -1: return j + 1
    if j == -1: return i + 1

    if s1[i] == s2[j]:
      memo[(i, j)] = dp(i - 1, j - 1) # skip
    else:
      memo[(i, j)] = min(
        dp(i, j - 1) + 1, # insert
        # insert the same char as s2[j] at s1[i]
        # then s2[j] is matched, move forward j, and cont comparing with i
        # ensure to add 1 to number of ops
        dp(i - 1, j) + 1, # delete
        # delete s1[i], move i forward, cont comparing with j
        dp(i - 1, j - 1) + 1, # replace
        # replace s1[i] with s2[j], then they are matched
        # move forward i, j and cont comparing
      )
    return memo[(i, )]
  # i, j init to the last index
  return dp(len(s1) - 1, len(s2) - 1)
```

DP array bottom-up version

```c
int minDistance(String s1, String s2) {
    int m = s1.length(), n = s2.length();
    int[][] dp = new int[m + 1][n + 1];
    // base case 
    for (int i = 1; i <= m; i++)
        dp[i][0] = i;
    for (int j = 1; j <= n; j++)
        dp[0][j] = j;
    // from the bottom up
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            if (s1.charAt(i-1) == s2.charAt(j-1))
                dp[i][j] = dp[i - 1][j - 1];
            else               
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i-1][j-1] + 1
                );
    // store the least editing distance of s1 and s2
    return dp[m][n];
}

int min(int a, int b, int c) {
    return Math.min(a, Math.min(b, c));
}
```

### Classic DP: Super Egg

Drop egg K from flour N, what's the worst number of tries (worst case, and at least tries) to find breakage floor F

- If K is unlimited, the least tries can be obtained by binary search
- **state** is K and N
- **choice** is which floor f to try next
- Formulation: a 2D dimensional DP array or DP function with two state parameters representing **state transition**; and a loop to traverse all choices, choose the optim to update state
- Time complexity of DP is #-subproblem * complexity of function itself: dp() is O(N) without recursive part; #-subproblem is total number of combination of different states - Cartesian product so O(KN); total is O(KN^2) and space is O(KN)

```python
def superEggDrop(K: int, N: int):

  memo = {}

  # Current state K eggs and N floors
  # Returns optimal result in this state
  def dp(K, N) -> int:
    # base case: N == 0, no throw needed; K == 1, all floors can only be searched linearly
    if K == 1: return N
    if N == 0: return 0

    if (K, N) in memo:
      return memo[(K, N)]

    res = float('INF')

    for i in range(1, N + 1):
      # Min number of eggs throwing in the worst case
      # max() because asking for worst case
      res = min(res, 
        max(
          dp(K - 1, i - 1), # broken
          dp(K, N - i) # not broken
        ) + 1 # throw once on the i-th floor
      )
    
    memo[(K, N)] = res
    return res

  return dp(K, N)
```

### Longest Common Subsequence

Similar to Edit Distance but with different state transition and choices

```python
def LCS(str1, str2):
  
  def dp(i, j):
    # base
    if i == -1 or j == -1:
      return 0
    if str1[i] == str2[j]:
      # found a char belongs to LCS (both in), keep going
      return dp(i - 1, j - 1) + 1
    else:
      # choice: pick the larger result of the two possible moves
      return max(dp(i - 1, j), dp(i, j - 1))

  # i,j init from backward
  return dp(len(str1) -1, len(str2) - 1)

# DP Table
def LCS(str1, str2):
  m, n = len(str1), len(str2)
  # build DP table and base case
  dp = [[0] * (n + 1) for _ in range(m + 1)]
  # state transition
  for i in range(1, m + 1):
    for j in range(1, n + 1):
      if str1[i - 1] == str2[j - 1]:
        dp[i][j] = 1 + dp[i - 1][j - 1]
      else:
        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

  return dp[-1][-1]
```

### Classic DP: Game Problem

There's a pile of stones as array of piles, piles[i] is how many stones in the i-th heap. Take turns with stones one pile a time but only the left or right one; after all stones taken, the last one who has more wins.

```python
piles = [1, 100, 3]

# whether it's 1 or 3, the 100 that's going to make the difference is going to be taken
# away by the back band, and the back hand is going to win.

# Design algo that returns the diff between final score of first hand and last hand,
# as in the example above, the first hand gets 4 points, the second 100, return -96
```

1. Define the meaning of DP table

The DP table can be defined in several ways, each lead to diff state transition equations; but as long as there's no logical hiccup, same result. 


```
Demo: piles = (3, 9, 1, 2)
Table: start\end 
start\end 0 - 1 - 2 - 3
0 (3,0) (9,3) (4,9) (11,4)
1 (   ) (9,0) (9,1) (10,2)
2 (   ) (   ) (1,0) (2,1)
3 (   ) (   ) (   ) (2,0)

dp[i][j].first repr highest score the first hand can get for this selection piles[i..j]
dp[i][j].second repr that of back hand highest score piles[i..j]

The demo starts at 0-index
dp[0][1].first == 9 means facing (3,9) the first hand will get 9 points
dp[1][3].second == 2 means facing (9,1,2) the second hand will get 2 points

Target: diff between final score of first hand and final score of second hand

  dp[0][n-1].first - dp[0][n-1].second


Set up:

range:
0 <= i < piles.length
i <= j < piles.length

n = piles.length
for 0 <= i < n:
  for j <= i < n:
    for who in (fir, sec):
      dp[i][j][who] = max(left, right)

dp[i][j].first = max(piles[i] + dp[i+1][j].second, piles[j] + dp[i][j-1].second)
dp[i][j].first = max( selection the rock pile on the far left, select far right)

I, as a first hand, faced piles[i..j] with 2 choices:
If choose far left, then face piles[i+1...j]
But when it comes to the other side, I become back hand
If choose far right, then face piles[i...j-1]
But in turn become back hand

if first hand select left:
  dp[i][j].second = dp[i+1][j].first
if first hand select right:
  dp[i][j].second = dp[i][j-1].first

Base case:

dp[i][j].first = piles[i]
dp[i][j].second = 0

range: 0 <= i == j < n

So the algo cannot simply traverse DP table row by row, BUT diagnoally

Solution:

class Pair {
    int fir, sec;
    Pair(int fir, int sec) {
        this.fir = fir;
        this.sec = sec;
    }
}

int stoneGame(int[] piles) {
    int n = piles.length;

    Pair[][] dp = new Pair[n][n];

    for (int j = i; j < n; j++)
      dp[i][j] = new Pair(0, 0);

    // base case
    for (int i = 0; i < n; i++) {
        dp[i][i].fir = piles[i];
        dp[i][i].sec = 0;
    }

    // traverse diagonally
    for (int l = 2; l <= n; l++) {
      for (int i = 0; i <= n - l; i++) {
          int j = l + i - 1;
          // first hand select left- or right-most pile
          int left = piles[i] + dp[i+1][j].sec
          int right = piles[j] + dp[i][j-1].sec
          // refer to state transition equation
          if (left > right) {
            dp[i][j].fir = left;
            dp[i][j].sec = dp[i+1][j].fir;
          } else {
            dp[i][j].fir = right;
            dp[i][j].sec = dp[i][j-1].fir;
          }
      }
    }
    Pair res = dp[0][n-1];
    return res.fir - res.sec;
}
```

### Strategies of Subsequence Problems

Generally asking for **longest subsequence** since the shortest is just a char. Almost certain need DP with O(n^2) time complexity Why? The possible subsequence is exponential.

Define DP array and find state transition relation.

First strategy: 1D DP array

```c
int n = array.length;
in[] dp = new int[n];

for (int i = 1; i < n; i++) {
    for (int j = 0; j < i; j++) {
        dp[j] = max|min(dp[i], dp[j] + ...)
      }
  }
```

- Demo: Longest increasing subsequence - DP array defined: dp[i] as the length of the required subsequence (LIS) within the subarray [0..i]
- Why LIS requires this? In line with induction method, and state transition relation can be found

Second strategy using 2D DP array

```c
int n = arr.length
int[][]dp = new dp[n][n];

for (int i = 0; j < n; i++) {
    for (int j = 0; j < n; i++) {
        if (arr[i] == arr[j])
          dp[i][j] = dp[i][j] + ...
      } else {
          dp[i][j] = max|min(...)
        }
  }
```

- In case of two string involved: the DP array is: dp[i][j] as the length of the rquired subsequence (LCS) within the subarray arr1[0..i] and arr2[0..j]
- In case of one string (longest palindrome subsequence) the dp[i][j] as the length of the required subsequence within array[i..j]

Longest Palindrome Subsequence

Why is it harder to solve for subsequence rather than substring? (discontinuous)

- Key: finding state transition relation requires inductive thinking. It is how we derive unknown parts from known results, making it easy to generalize and discover the state transition relation.
- To find dp[i][j], suppose you have already got the result of the subproblem dp[i+1][j-1] (the length of the LPS in 1-index inner range), can you find a way to calculate the value of dp[i][j] (the length of the target range)?
- It depends on the chars of s[i] and s[j]: if equal, then LPS would be these two chars plus current length; else, they cannot appear at the same time in LPS of s[i..j].
Therefore we add them separately to s[i+1..j-1] to see which substring produces a longer palindrome subsequence.
- At this point, the state transition equation is derived. Per definition of DP array, we need dp[0][n-1] as the final result of string of length

```
i i+1       j-1 j  
? b   x a b y   ?

dp[i+1][j-1] = 3

if [i] == [j]:
  // for example: 'c'
  // then result = 3 + 2 = 5
  dp[i][j] = dp[i+1][j-1] + 2;

else:
  // for example 'a' and 'b'
  // consider [i][j-1]
  // consider [i+1][j]
  // choose the longer palindrome subsequence from the two
  dp[i][j] = max(dp[i+1][j], dp[i][j-1]);
```

- Base case: if only one char, LPS is 1, or dp[i][j] = 1 (i == j)
- Since i must be less or equal to j, for those locations where i > j, they are null or 0
- Per state transition equation, need to know 3 states to arrive at the result

```
DP array

i\j

1       X
0 1
0 0 1
0 0 0 1
0 0 0 0 1

In order to guarantee that before each calcu of dp[i][j], 
the values in the left, down, right direction have been calcu,
we can only only traverse it diagnoally or reversely.

1 \ ->
0 1 \
0 0 1 \
0 0 0 1 \
0 0 0 0 1

or

------>
 -----> ^
  ----> |
   ---> |
```

```c
// Traverse reversely
int LPS(string s) {
    int n = s.size();
    // DP array all init to 0
    vector<vector<int>> dp(n, vector<int>(n, 0));
    // base 
    for (int i = 0; i < n; i++)
      dp[i][i] = 1;
    // Reverse traversal to ensure correct state transition
    for (int i = n - 1; i >= 0; i--) {
        for (int j = 1 + 1; j < n; j++) {
            if (s[i] == s[j])
              dp[i][j] = dp[i + 1][j - 1] + 2;
            else
              dp[i][j] = max(dp[i+1][j], dp[i])
        }
    }
  // return result
  return dp[0][n - 1];
}
```

### 4 Keys Keyboard

Problem: special 4-key keyboard
- Key1 (A): Print 'A'
- Key2 (C-A): select the whole screen
- Key3 (C-C): copy selection to buffer
- Key4 (C-V): print buffer appending to previous stdout

Can only press key N times and find out the max numbers of 'A' you can print.

Framework thinking: seems a enumeration and finding best - DP problem. Enumerating all choices at each transition and find the best outcome.

DP Key: find STATE and CHOICE. 

CHOICE is simple, the 4 keys. STATE? What info needed to know to break original problem into subproblem.

- First state is the remaining number of times - n
- Second state is number of char 'A' on current screen - a_num
- Third state is number of char 'A' still in the clipboard or buffer - copy

Hence base: when number of remaining n = 0, a_num is the answer

```
dp(n - 1, a_num + 1, copy) #[A]

# Press 'A', subtract 1 from n

dp(n - 1, a_num + copy, copy) #[C-V]

# paste, add to a_num with copy

dp(n - 2, a_num, a_num) #[C-A] & [C-C]

# select all and copy to buffer (Note used together)
# a_num (the 'A' in buffer) now becomes printed 'A' on current screen
```

```python
def maxA(N: int) -> int:

  # memo
  memo = {}
  
  def dp(n, a_num, copy):
    # base
    if n <= 0: return a_num;

    if (n, a_num, copy) in memo:
      return memo[(n, a_num, copy)]

    memo[(n, a_num, copy)] = max(
      dp(n - 1, a_num + 1, copy),
      dp(n - 1, a_num + copy, copy),
      dp(n - 2, a_num, a_num),
    )

    return memo[(n, a_num, copy)]

  return dp(N, 0, 0)
```

This method still search for a large number; the time complexity challenge is not easy. No matter what it is, now write DP-function as DP-array.

```c
dp[n][a_num][copy]
// Total number of states (spatial complexity) is the volume of this 3D array

// Hard to compute max number of a_num and copy. The lowest complexity is O(N^3)

// Now increase the complexity of induction but reduce spatial complexity.
// Only keep one state - n
// Idea: there must be only two cases of key sequence corresponding to the optimal result
// - Either keeps pressing A: A, A, ... A (when N is smaller)
// - Or this: A, A, ..., C-A, C-C, C-V, C-V, ... C-V (mostly when N is larger) (some mathematical rules involved)
// So the last Key is either A or C-V. (2 cases)

int[] dp = new int[N + 1];
// dp[i] the max number of A
for (int i = 0; i <= N; i++)
  dp[i] = max(
    // Press [A]
    // Press [C-V]
  )

// For the case of [A], it is actually a new 'A' printed of state i-1
dp[i] = dp[i - 1] + 1;

// But for C-V, need to consider C-A and C-C
// Use j as the starting point for these sequence of C-ACV, then the two ops before j should be C-A and C-C

public int maxA(int N) {
  int[] dp = new int[N + 1];
  dp[0] = 0;
  for (int i = 1; i <= N; i++) {
    // [A]
    dp[i] = dp[i - 1] + 1;

    for (int j = 2; j < i; j++) {
      // C-A & C-C -> dp[j-2], Paste i-j times
      // There are { dp[j-2] * (i-j+1) } number of 'A' printed
      dp[i] = Math.max(dp[i], dp[j - 2] * (i - j + 1));
    }
  }
  return dp[N];
}

// Time O(N^2) and space O(N)
```

- The issue with the first exhaustive approach is that some of the sequence (C, A, C, A) is not optimal, even if the overlap is cached, the search operation still remain

### DP Optimal Substructure

- optimal substructure exists outside DP, those don't have overlapping subproblem
- optimal substructure needs subproblem to be independent
- Trick is: convert the problem e.g. max score difference in school (knowning diff in class), write a piece of brute-force code

```
int result = 0;
for (Student a : school) {
    for (Student b : school) {
        if (a is b) continue;
        result = max(result, |a.score - b.score|);
    }
}
return result;
```

- DP is nothing more than solving the overlapping subproblems (with optimal substructure conversion)

```
// Another example, find max in binary tree
int maxVal(TreeNode root) {
    if (root == null)
      return -1;
    int left = maxVal(root.left);
    int right = maxVal(root.right);
    return max(root.val, left, right);
  }
```

- Above is NOT a DP problem, hence the fact that optimal-substructure is not unique to DP
- Hence any problem concerned with finding optimal (max,min) DP can be one of the solution
- The process of finding the optimal substructure is actually the process of verifying the correctness of state transition equation. Brute-force solution is a sign, overlapping is for optimization

Traversal Order of DP array

```
// sometimes forward
int[][] dp = new int[m][n];
for (int i = 0; i < m; i++)
  for (int j = 0; j < n; j++)
    // calcu dp[i][j]

// backward
for (int i = m - 1; i >= 0; i--)
  for (int j = n - 1; j >= 0; j--)
    // calcu dp[i][j]

// diagnoally
for (int l = 2; l <= n; l++)
  for (int i = 0; i <= n - l; i++)
    int j = l + i - 1;
    // calcu dp[i][j]
```

- During traversal, all the required states must have been calculated
- The final point of traversal must be the point where the result is stored

```
// Edit distance: by definition, bases is dp[..][0] and dp[0][..];
// final result is dp[m][n]
// transition dp[i][j] derived from dp[i-1][j], dp[i][j-1], dp[i-1][j-1]
// hence forward traversal: 
// dp[i-1][j], dp[i][j-1] (left, top) since top-left is base case

// LPS is either; 

// All depends on order of calculation towards target
```

### Longest Increasing Subsequence

To develop general technique for designing DP: mathematical induction
Often easier to find O(N^2) and then with binary search to O(N log N)

Problem: given unsorted array of integers, find the length of LIS

Induction: assuming a conclusion is true when K < N, (base?) then think of a way to prove that this conclusion still hold when K = N

Similarly, when designing DP array, assuming that `dp[0..i-1]` found, then ask how to find `dp[i]`

First, define DP array: `dp[i]` represents the length of LIS ending with `nums[i]` NOTE: this is not the range but the subsequence of exactly from 0 to i inclusive in considering LIS

```
index
0 1 2 3 4
nums
1 4 3 4 2

dp[3] = 3 // 1 -> 3 -> 4
dp[4] = 2 // 1 -> 2

// This definition means the result is the max value in DP array
int res = 0;
for (int i = 0; i < dp.size(); i++)
  res = Math.max(res, dp[i]);

// How to design algo to compute each dp[i] correctly?
// This is the highlight of DP - to think about state transition via induction

// How to derive dp[5] after knowing all the values of dp[0..4]

dp
1 2 2 3 2 ?

// Per this DP array, it's to find the LIS ending with nums[5], which = 3 since only
// need to find the previous subsequences with a smaller end than 3
// after connecting 3 to the end to form a new increasing subsequence, the new subsequence
// length will be incremented by 1

for (int j = 0; j < i; j++)
  if (nums[i] > nums[j])
    dp[i] = Math.max(dp[i], dp[j] + 1)

// Above can find dp[5], how about previous i
// outer loop
for (int i = 0; i < numbs.length; i++)
  // above

// For a string, every subsequence length is at least 1, so init array with 1

// Full
public int lengthOfLIS(int[] nums)
  int[] dp = new int[nums.length];

  Arrays.fill(dp, 1);

  for (int i = 0; i < nums.length; i++)
    for (int j = 0; j < i; j++)
      dp[i] = Math.max(dp[i], dp[j] + 1)

  int res = 0;
  for (int i = 0; i < dp.length; i++)
    res = Math.max(res, dp[i])

  return res
```

- Recap: first define DP array - the most important step
- Then use induction, given values of `dp[0..i-1]`, find a way to compute `dp[i]`
- If at this point the step cannot be completed, check DP array definition, or info in array not enough to introduce the next answer, thus a DP of 2D or 3D required
- Think base case

Binary search solution

- Hard to devise, but LIS is related to card game called patience game
- Dealing cards into increasing piles, each lower card is put onto a new pile; the resulting number of piles is the answer of length of LIS
- Then apply binary search in placing cards

```c
public int lengthOfLIS(int[] nums) {
    int[] top = new int[nums.length];
    // Initialize the number of piles
    int piles = 0;
    for (int i = 0; i < nums.length; i++) {
        // play cards to be handled
        int poker = nums[i];

        /***** binary search *****/
        int left = 0, right = piles;
        while (left < right) {
            int mid = (left + right) / 2;
            if (top[mid] > poker) {
                right = mid;
            } else if (top[mid] < poker) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        /*********************************/

        // create a new pile and put this card into it
        if (left == piles) piles++;
        // put the card on the top
        top[left] = poker;
    }
    // The number of piles represents the length of LIS
    return piles;
}
```

### KMP Algo

Knuth-Morris-Pratt Algo for string matching.

Algo-4, the name of the array used in original code is Determining the Finite State Machine.

What's brute-force and KMP algo.

```c
// brute-froce
int search(String pat, String txt)
  int M = pat.length
  int N = txt.length

  for (int i = 0; i <= N - M; i++)
    int j;
    for (j = 0; j < M; j++)
      if (pat[j] != txt[i+j])
        break;
      // pat all matches
      if (j == M) return i;
  // pat substring does not exist in txt
  return -1;

// Time complexity O(MN) and space O(1)
// Main problem is that if there are many repeated chars in string, wasteful
// txt = 'aaacaaab' pat = 'aaab'
// no 'c' in pat and not necessary to roll back the pointer i

// KMP takes space to record some info
// It knows that the 'a' before 'b' is matched, so only need to compare if 'b' is matched
// every time

// KMP never rolls back point i of txt and does not go back (does not scan txt repeatedly), 
// but uses the info stored in DP array to move pat to the correct position and continue
// Time O(N) and space is used for time

// Hard part is how to compute the info in DP-array. How to move the pointer of pat correctly
// based on this info? This requires Determinant Finite State Machine
// It's exactly the same as DP-array

// Note only to compute DP-array related to pat, regardless of txt

// txt
//     i
// a a a c a a a b
//       |
// pat   |
// a a a b
//     j |
//       |
// move ->
//       |  a a a b
//       j
//
// NOTE: j should not be seen as index, its meaning should be more of state

public class KMP
 private int[][] dp;
 private String pat;

 public KMP(String pat)
  this.pat = pat;
  // Build DP array from pat - O(M) time

  public int search(String txt)
    // Match txt with DP-array - O(N) time
    // Note DP-array is the same regardless of txt


// Why State Machine
// pat matching as a state transition - pat = 'ABABC'
// 0 -> 1 -> 2 -> 3 -> 4 -> 5
//   A    B    A    B    C
// 
// Index is state of matching done
// Its transition behaves differently in different states
// e.g. now matches state 4; sees 'A', it should transition to state-3
// 4 sees A -> 3
// 4 sees C -> 5
// 4 sees B -> 0

// From State Machine to DP-array
// At any step, need 2 states: current j and current char

// 2D array for DP-array
dp[j][c] = next
0 <= j < M // the current state of the table
0 <= c < 256 // ASCII char encountered
0 <= next <= M // next state

dp[4]['A'] = 3 // means current state 4, if 'A', pat should go state 3

public int search(String txt)
  int M = path.length();
  int N = txt.length();
  // init state of pat is 0
  int j = 0;
  for (int i = 0; i < N; i++)
    // current state j, char is txt[i]
    // which state pat go to
    j = dp[j][txt.charAt(i)];
    // if termination state is reached, index at beginning of match returned
    if (j == M) return i - M + 1;

  return -1;

// Building the DP-array or state machine
for 0 <= j < M:
  for 0 <= c <= 256;
    dp[j][c] = next

// if c == pat[j], state move forward next = j + 1 (state advance)
// else, roll back / unchange (state reset)
// But how? Define Shadow State repr by X, with same prefix as current state
// e.g.
// when j at 4, X at 2, because they share the same prefix 'AB'
// thus when j to state-reset, can use X state transition diagram to get recent reset position
// e.g. if j sees 'A', first the state can only be advanced if 'C', so it can only restart the
// state now; state j will delegate this char to state X, dp[j]['A'] = dp[X]['A']
// Why? because since now needs to roll back, KMP wants to rollback as little as possible; then j
// can ask X with the same prefix as itself. If X meets 'A' and can perform state-advance, then it
// will be transferred, because it will have the least rollback
// e.g. if j meets 'B', state X cannot be state-advance and need to rollback; j just needs to roll
// back in the direction of X
// How does X know that when it meets 'B', it will fall back to state-0? Because X always follows
// behind j, its shifts has been computed before - DP array keeps cache

int X // shadow state
for 0 <= j < M:
  for 0 <= c < 256:
    if c == pat[j]:
      // state advance
      dp[j][c] = j + 1;
    else:
      // state-reset
      // delegate X to compute reset position
      dp[j][c] = dp[X][c]

// How did the shadow-state X get?
public class KMP {
    private int[][] dp;
    private String pat;

    public KMP(String pat) {
        this.pat = pat;
        int M = pat.length();
        // dp[state][character] = next state
        dp = new int[M][256];
        // base - only when pat[0] is met can state advance from 0 to 1
        dp[0][pat.charAt(0)] = 1;
        // shadow state X init 0
        // it keeps being updated as j advances
        int X = 0;
        // current state j at 1
        for (int j = 1; j < M; j++)
          for (int c = 0; c < 256; c++)
            if (pat.charAt(j) == c)
              dp[j][c] = j + 1;
            else
              dp[j][c] = dp[X][c];
          // update shadow state
          // the current state is X, the char pat[j] is met,
          // updating X similar to updating status j in search()
          // j = dp[j][txt.charAt(i)]
          // state X is always one state behind state j, with the same longest prefix
          // so the new X state should go to (similar to search where j goes to a position when
          // current j meets txt.charAt(i)) where current X meets pat.charAt(j) - the new char 
          // that state of j changed to (to keep shadowing j)
          X = dp[X][pat.charAt(j)];
```

### House Robber problem

### Stock Buy-Sell

## Data Structure

### Binary Heap and Priority Queue

### LRU Cache

### Collection of Binary Search Operations

### Special DS: Monotonic Stack

### Design Twitter

### Reverse Part of Linked List via Recursion

### Best Algo Book

### Queue-Stack and Stack-Queue

## Algo Thinking

### Ideas

### Framework of Backtracking Algo

### Binary Search in Detail

### Double Pointer

### TwoSum Problems

### Divide Complicated Problem: Implement a Calculator

### Prefix Sum Skill

### FloodFill Algo

### Interval Scheduling: Interval Merging

### Interval Scheduling: Intersections of Intervals

### String Multiplication

### Pancake Soring Algo

### Useful Bit Manipulation

### Russian Doll Envelopes Problem

### Recursion in Detail

### Backtracking to Subset/Permutation/Combination

### Several Counter-intuitive Probability Problems

### Shuffle Algo

## High Frequency Interview Questions

## Common Knowledge

### Linux Process and Thread

### Linux Shell

### Cookies and Session

### Cryptography

### Online Practice

Git - https://learngitbranching.js.org

SQL - sqlzoo.net

