## Algo Thinking

### Ideas

1. What is it? (stack - go read textbook)
2. What is it for? (practice questions)

- Read a chapter and build your own. DA is just combo of array and linked-list with APIs add, delete, search, modify
- Practice means big data and pattern recognition (framework) maze -> graph search -> tree search -> BST

### Framework of Backtracking Algo

Think in traversal of a decision tree.

- Path: the selection made
- Selection List: selection to make next
- End Condition: terminating condition when reaching leaf

Demo: Permutation and N Queen Problem

```
result = []
def Backtrack(Path, Selection List):
  if meet the End Condition:
    result.add(Path)
    return

  for selection in Selection List:
    select
    backtrack(Path, Selection List)
    deselect
```

- The core is the recursion in the for loop. It makes a selection before the recursion call and undo the selection after the recursion call

Permutation

For N unique numbers, the number of full permutation is `N!` (assuming unique)

Idea: `[1, 2, 3]`

1. Fix the first number to 1
2. Then second number can be 2
3. If the second is 2, then third can only be 3
4. Then you can change the second number to 3 and the third number can only be 2
5. Then you can only change the first place, and repeat 2-4

This is exactly backtracking tree. Or a decision tree. At 2, 2 is the Path behind, 1,3 is the Selection List to make; end condition is to traverse to leaf (i.e. when List is empty)

`backtrack()` defined is like a pointer - exhaustively walk the tree and maintain the attributes of each node (Path, Selection List) correctly. End condition means full permutation

```java
// Tree traversal
void traverse(TreeNode root) {
  for (TreeNode child : root.children)
    // Ops needed for pre-order traversal
    traverse(child);
    // Ops needed for post-order traversal
}

// Pre/Post-Order are just two very useful time points
// Pre-Order is executed at the time point before entering a node, while Post-Order is executed at the time point after leaving a node
// Given Path and Selection List are attributes of each node - to keep the integrity, must do something
// Pre-Order: select ([1] -> [2,3])
// Post-Order: unselect ([] -> [1,2,3])
```

```
for selection in Selection List:
 // Preorder - select
 Remove this selection from List
 Path.add(selection)

 // traverse
 backtrack(Path, Selection List)

 // Postorder - deselect
 Path.remove(selection)
 Add the selection to List
```

```java
List<List<Integer>> res = new LinkedList<>();

/* The main method, enter a set of unique numbers and return their full permutations */
List<List<Integer>> permute(int[] nums) {
    // record Path
    LinkedList<Integer> track = new LinkedList<>();
    backtrack(nums, track);
    return res;
}

// Path: recorded in track
// Seletion List: those elements in nums that do not exist in track
// End Condition: all elements in nums appear in track
void backtrack(int[] nums, LinkedList<Integer> track) {
    // trigger the End Condition
    if (track.size() == nums.length) {
        res.add(new LinkedList(track));
        return;
    }

    for (int i = 0; i < nums.length; i++) {
        // exclude illegal seletions
        if (track.contains(nums[i]))
            continue;
        // select
        track.add(nums[i]);
        // enter the next level decision tree
        backtrack(nums, track);
        // deselect
        track.removeLast();
    }
}

// nums and track to deduce the current selection list

// Time O(N); better ways via exchanging elements are harder to comprehend
// Regardless, time cannot be less than O(N!) because exhaustion of the entire
// tree is unavoidable
// This is also a feature of backtracking - unlike DP having overlapping subproblems which
// can be optim, this is purely exhaustion, time high time complexity
```

```java
vector<vector<string>> res;

/* Enter board length n, return all legal placements */
vector<vector<string>> solveNQueens(int n) {
    // '.' Means empty, and 'Q' means queen, initializing the empty board.
    vector<string> board(n, string(n, '.'));
    backtrack(board, 0);
    return res;
}

// Path:The rows smaller than row in the board have been successfully placed the queens
// Seletion List: all columns in 'rowth' row are queen's seletions
// End condition: row meets the last line of board(n)
void backtrack(vector<string>& board, int row) {
    // trigger the End Condition
    if (row == board.size()) {
        res.push_back(board);
        return;
    }

    int n = board[row].size();
    for (int col = 0; col < n; col++) {
        // exclude illegal seletions
        if (!isValid(board, row, col)) 
            continue;
        // select
        board[row][col] = 'Q';
        // enter next row decision
        backtrack(board, row + 1);
        // deselect
        board[row][col] = '.';
    }
}

/*Is it possible to place a queen on board [row] [col]? */
bool isValid(vector<string>& board, int row, int col) {
    int n = board.size();
    // Check if share the same column
    for (int i = 0; i < n; i++) {
        if (board[i][col] == 'Q')
            return false;
    }
    // Check if share the same right diagonal
    for (int i = row - 1, j = col + 1; 
            i >= 0 && j < n; i--, j++) {
        if (board[i][j] == 'Q')
            return false;
    }
    // Check if share the same left diagonal
    for (int i = row - 1, j = col - 1;
            i >= 0 && j >= 0; i--, j--) {
        if (board[i][j] == 'Q')
            return false;
    }
    return true;
}

// Each row (akin to current Path) and column (akin to possible columns at the row)

// While Gauss spent his whole life on N = 8 without a solution, the time is O(N ^ (N+1)) worst
// If N = 10, the computation is rather huge

// Returns true after finding an answer
bool backtrack(vector<string>& board, int row) {
    // Trigger End Condition
    if (row == board.size()) {
        res.push_back(board);
        return true;
    }
    ...
    for (int col = 0; col < n; col++) {
        ...
        board[row][col] = 'Q';

        if (backtrack(board, row + 1))
            return true;

        board[row][col] = '.';
    }

    return false;
}

// This will terminate once an answer found
```

N Queen

Place N non-attacking queens on a N x X chessboard. A solution requires that no two queens share the same row, column, or diagonal.

This is similar to full permutation - a decision tree each layer repr each row and selection that each node can make is to place a queen on any column of the row

### Binary Search in Detail

Basic framework

```java
int binarySearch(int[] nums, int target) {
    int left = 0, right = ...;

    while(...) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            ...
        } else if (nums[mid] < target) {
            left = ...
        } else if (nums[mid] > target) {
            right = ...
        }
    }
    return ...;
}
```

- A technique for analyzing binary search is: do not appear else, but write everything clearly with else if, so that all details can be clearly displayed
- Be careful of details
- Prevent overflow when calculating mid (too large a direct addition causes overflow)

First, find a number

```java
int binarySearch(int[] nums, int target) {
    int left = 0; 
    int right = nums.length - 1; // attention

    while(left <= right) {
        int mid = left + (right - left) / 2;
        if(nums[mid] == target)
            return mid; 
        else if (nums[mid] < target)
            left = mid + 1; // attention
        else if (nums[mid] > target)
            right = mid - 1; // attention
    }
    return -1;
}
```

- Why `<=` in condition of while loop? - since init assignment of `right` is `nums.length -1`, which is index of last element, not `nums.length`
- In this aglo, the interval `[left, right]` closed at both ends. This interval is actually the interval for each search.
- Stop condition is when target found: `if (nums[mid] == target) return mid`
- But if not found, need to terminate while loop and return -1; (when search interval is empty)
- Condition `while (left <= right)`, or `left == right + 1` in the form of an interval `[right + 1, right]`, so the interval is empty at this time, since no number is possible
- Condition `while (left < right)` or `left == right` in `[left, right]` (2,2) this is the interval not empty and there is number 2, but at this time the while loop stops. i.e. this is ommitted and index 2 is not searched, this is the wrong condition
- Why `left = mid + 1; right = mid - 1`? As opposed to `right = mid, left = mid` Without these additions, subtractions - search interval is closed at both ends, `[left, right]`, so when finding index `mid != target` where to search next? - `[left, mid - 1], or [mid + 1, right]` - because `mid` has already been searched, hence removed from search interval
- Flaws - e.g. `[1,2,2,2,3] target = 2` index returned 2. But if to get left border of `target`, which is index 1, or right, 3, then this algo cannot handle it - a common requirement

Binary search to find the left border

```java
int left_bound(int[] nums, int target) {
    if (nums.length == 0) return -1;
    int left = 0;
    int right = nums.length; // attention
    // search interval [left, right)
    
    // the condition of termination is left == right, or [left, left) is empty
    // is the right not nums.length - 1? why left open
    while (left < right) { // attention
        int mid = (left + right) / 2;
        if (nums[mid] == target) {
            right = mid;
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid; // attention
        }
    }
    return left;
}
```

In sum:

- do not use `else` but expand all `if else`
- pay attention to termination condition of search interval and while
- if need to define left and right search interval to search left and right boundaries, only need to modify it when `nums[mid] == target` and need to subtract one when searching the right side
- if search interval is unified to be closed at both ends, it's easy to recall, as long as change code and return logic at `nums[mid] == target`, it is advised to take a small book as a binary search template

https://labuladong.gitbook.io/algo-en/iii.-algorithmic-thinking/detailedbinarysearch

### Double Pointer

First, common algo of fast-slow double pointers - usually init to point to head node of linked list

1. Determine whether the linked list contains a ring

Used for singly linked to determine if there's a ring.

```java
boolean hasCycle(ListNode head) {
  while (head != null)
    head = head.next
  return false;
}

// but if there's a ring, the pointer will end up in endless loop

// classic solution is to use two pointers, one running fast;
boolean hasCycle(ListNode head)
  ListNode fast, slow;
  fast = slow = head;
  while (fast != null && fast.next != null)
    fast = fast.next.next; // twice faster
    slow = slow.next
    
    if (fast == slow) return true;

  return false;
```

2. Knowing that the linked list contains a ring, return to the starting position of the ring

```java
ListNode detectCycle(ListNode head) {
    ListNode fast, slow;
    fast = slow = head;
    while (fast != null && fast.next != null) {
        fast = fast.next.next;
        slow = slow.next;
        if (fast == slow) break;
    }
    // The above code is similar to the hasCycle function
    slow = head;
    while (slow != fast) {
        fast = fast.next;
        slow = slow.next;
    }
    return slow;
}


```

Second, the common algo of left and right pointer - two index pointer to 0 and length - 1

1. binary search
2. Two sum - given an array of integers, return indices of two numbers such that they add up to a target; may assume that each input would have unique solution and may not use the same element twice.

```
[2, 7, 11, 15], target = 5

[0, 1] because nums[0] + nums[1] = 2 + 7 = 9
```

As long as array ordered, should think of the two pointer technique - similar to binary search and can adjust the size of 'sum' by adjusting 'left' and 'right'

```java
int[] twoSum(int[] nums, int target)
  int left = 0, right = nums.length - 1;
  while (left < right)
    int sum = nums[left] + nums[right];
    if (sum == target)
      // index required for the question starts at 1
      return new int[]{left + 1, right + 1}
    else if (sum < target)
      left++; // make 'sum' bigger
    else if (sum > target)
      right--; // make sum smaller

  return new int[]{-1, -1};
```

3. reverse array

```java
void reverse(int[] nums) {
    int left = 0;
    int right = nums.length - 1;
    while (left < right) {
        // swap(nums[left], nums[right])
        int temp = nums[left];
        nums[left] = nums[right];
        nums[right] = temp;
        left++; right--;
    }
}
```

4. sliding window

See Algo Thinking.

### TwoSum Problems

TwoSum I - given an array `nums` and `target`, return indices of two numbers such that they add up to `target` assuming unique solution.

```java
// simple exhaustive search
int[] twoSum(int[], nums, int target)
  
  for (int i = 0; i < nums.length; i++)
    for (int j = i + 1; j < nums.length; j++)
      if (nums[j] == target - nums[i])
        return new int[] { i, j };

  return new int[] { -1, -1 };

// time O(n^2) and space O(1)

// hash table to reduce time complexity
int[] twoSum(int[] nums, int target)
  int n = nums.length;
  index<Integer, Integer> index = new HashMap<>();
  // build hash table: elem are mapped to their corresponding indices
  for (int i = 0; i < n; i++)
    index.put(nums[i], i);

  for (int i = 0; i < n; i++)
    int other = target - nums[i];
    // IF 'other' exists and it is not nums[i]
    if (index.containsKey(other) && index.get(other) != i)
      return new int[] { i, index.get(other) };

  return new int[] { -1, -1 };

// query time of hash table is O(1) hence reduces to O(N)
// but space increases to O(N)

// The main idea is the use of hash table property to solve two-sum
```

TwoSum II

```java
// modify into class
class TwoSum
  Map<Integer, Integer> freq = new HashMap<>();

  public void add(int number);
    freq.put(number, freq.getOrDefault(number, 0) + 1);

  public boolean find(int value);
    for (Integer key : freq.keySet())
      int other = value - key;
      if (other == key && freq.get(key) > 1)
        return true;
      if (other != key && freq.containsKey(other));
        return true;

    return false;

// if array ordered, double-pointer solution
int[] twoSum(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left < right) {
        int sum = nums[left] + nums[right];
        if (sum == target) {
            return new int[]{left, right};
        } else if (sum < target) {
            left++; // Make sum bigger
        } else if (sum > target) {
            right--; // Make sum smaller
        }
    }
    // If no such two numbers exists
    return new int[]{-1, -1};
}
```

### Divide Complicated Problem: Implement a Calculator

1. Convert string to integer - 'how to convert a positive integer as string into integer'

```java
string s = "458";

int n = 0;
for (int i = 0; i < s.size(); i++)
  char c = s[i];
  n = 10 * n + (c - '0');

// n is now equal to 458
```

2. Processing addition and subtraction

- if input formula only contains addition and subtraction, no space (1 - 12 + 3)
- `+` prepend
- combine an operator and a number into a pair, `+1, -12, +3` put on stack
- summing all numbers in stack

```java
int calculate(string s)
  stack<int> stk;
  int num = 0;
  // Record the sign before num, init to +
  char sign = '+';
  for (int i = 0; i < s.size(); i++)
    char c = s[i];
    // If number, assign it continuously to num
    if (isdigit(c))
      num = 10 * num + (c - '0');
    // Else, it must be the next symbol
    // the preivous numbers and symbols should be on the stack
    if (!isdigit(c) || i == s.size() - 1)
      switch (sigh)
        case '+':
          stk.push(num); break;
        case '-':
          stk.push(-num); break;
      // Update symbol to current symbol and clear the number
      sign = c;
      num = 0;
  // sum al results in the stack is the answer
  int res = 0;
  while (!stk.empty())
    res += stk.top();
    stk.pop();

  return res;

// Full
def calculate(s: str) -> int:

    def helper(s: List) -> int:
        stack = []
        sign = '+'
        num = 0

        while len(s) > 0:
            c = s.pop(0)
            if c.isdigit():
                num = 10 * num + int(c)
            # Meet the left parenthesis and start recursive calculation of num
            if c == '(':
                num = helper(s)

            if (not c.isdigit() and c != ' ') or len(s) == 0:
                if sign == '+':
                  stack.append(num)
                elif sign == '-':
                  stack.append(-num)
                elif sign == '*':
                  stack[-1] = stack[-1] * num
                elif sign == '/':
                  stack[-1] = int(stack[-1] / float(num))
                num = 0
                sign = c
            # Return recursive result when encountering right parenthesis
            if c == ')': break
        return sum(stack)

    return helper(list(s))
```

### Prefix Sum Skill

Given array of integers and int k, find out num of sub-arrays which sums to k

- brute-force find all sub-arrays, sum up and compare with k
- tricky part is how to find sub-array fast
- Prefix sum is O(1) to find sum

1. What is Prefix Sum - create another array to store the sum of prefix for pre-processing

```java
int n = nums.length;
int[] preSum = new int[n + 1];
preSum[0] = 0;
for (int i = 0; i < n; i++)
  preSum[i + 1] = preSum[i] + nums[i];

// index   0 1 2 3  4 5  6
// nums    3 5 2 -2 4 1  
// preSum  0 3 8 10 8 12 13 
// Idea: sum of nums[i..j] = preSum[j+1] - preSum[i]

int subarraySum(int[] nums, int k)
  int n = nums.length;
  int[] sum = new int[n + 1];
  sum[0] = 0;
  for (int i = 0; i < n; i++)
    sum[i + 1] = sum[i] + nums[i];

  int ans = 0;
  for (int i = 1; i <= n; i++)
    for (int j = 0; j < i; j++)
      if (sum[i] - sum[j] == k)
        ans++;

  return ans;

// time O(N^2) space O(N)
// sub-optim but it's cool trick to reduce time further
```

2. Optim Solution

- looking at the double loop, what does the inner loop do? - compute how many j can make a difference of `sum[i]` and `sum[j]` to be k; whenever we find such j, we'll increment result by 1

```java
// reorganize if condition 
if (sum[j] == sum[i] - k)
  ans++;

// Idea: to record down how many sum[j] = sum[i] - k such that we can update the result
// directly instead of having inner loop - hash table to record both prefix sums and the
// frequency of each prefix sum
int subarraySum(int[] nums, int k) {
    int n = nums.length;
    // map：prefix sum -> frequency
    HashMap<Integer, Integer> 
        preSum = new HashMap<>();
    // base case
    preSum.put(0, 1);

    int ans = 0, sum0_i = 0;
    for (int i = 0; i < n; i++) {
        sum0_i += nums[i];
        // this is the prefix sum we want to find nums[0..j]
        int sum0_j = sum0_i - k;
        // if it exists, we'll just update the result
        if (preSum.containsKey(sum0_j))
            ans += preSum.get(sum0_j);
        // record the prefix sum nums[0..i] and its frequency
        preSum.put(sum0_i, 
            preSum.getOrDefault(sum0_i, 0) + 1);
    }
    return ans;
}

//              i
//   3 5 2 -2 4 1
// 0 3 8 10 8 12 13 (sum0_i)
// k = 5 need to find prefix sum of 13 - 5 = 8
// time O(N)

// Summary
// very useful in dealing with diff in array intervals
// e.g. if compute percentage of each score interval among all students
int[] scores; // all students score
// full score is 150
int[] count = new int[150 + 1]
// record how many students at each score
for (int score : scores)
  count[score]++
// build prefix sum
for (int i = 1; i < count.length; i++)
  count[i] = count[i] + count[i-1];

// after, for any given interval, to find how many students fall in this interval by computing
// diff of prefix sums fast. 
// BUT for more complex problems, need more.
// Hash table used to eliminate unnecessary loop
```

### FloodFill Algo

Application: color-filling; Minesweeper - the process of expanding on clicking a tile; puzzle-matching like Candy Crush used to remove blocks of same color.

1. Framework

- all above examples can be abstracted as 2D array, which can be further abstracted as graph
- Problem becomes graph traversal, like N-ary tree.

```java
void fill(int x, int y)
  fill(x - 1, y); // up
  fill(x + 1, y); // down
  fill(x, y - 1); // left
  fill(x, y + 1); // right
```

- This framework can resolve all problems of traversing 2D array - DFS or quaternary (4-ary) tree traversal

LeetCode: image repr as 2D-array of integers, each a pixel value (0, 65535); given (sr, sc) coordinate, repr the starting pixel (row, column) of the flood fill, and a pixel value `newColor`, flood fill the image.
To perform it, consider starting pixel, plus any pixels connected 4-directionally to the starting pixel of the same color as the starting pixel, plus any pixels connected 4-directionally to those pixels (aslo wit the same color as starting pixel), and so on. Replace the color of all said pixels with newColor; return modified image.

```
image = [
  [1,1,1],
  [1,1,0],
  [1,0,1],
]
sr = 1, sc = 1, newColor = 2
output = [
  [2,2,2],
  [2,2,0],
  [2,0,1],
]

From the center, (sr, sc) = (1, 1), all pixels connected by a path of the same color
as starting pixel are colored with the new color.
Note bottom corner is not 2, because it is not 4-directionally connected to starting pixel
]
```

```java
int[][] FloodFill(int[][] image, int sr, int sc, int newColor)

  int origColor = image[sr][sc];
  fill(image, sr, sc, origColor, newColor);
  return image;

void fill(int[][] image, int x, int y, int origColor, int newColor)
  // OUT: out of index
  if (!inArea(image, x, y)) return;
  // CLASH: meet other colors, beyond the area of origColor
  if (image[x][y] != origColor) return;
  image[x][y] = newColor;

  fill(image, x, y + 1, origColor, newColor);
  fill(image, x, y - 1, origColor, newColor);
  fill(image, x - 1, y, origColor, newColor);
  fill(image, x + 1, y, origColor, newColor);

boolean inArea(int[][] image, int x, int y)
  return x >= 0 
      && y < image.length
      && y >= 0
      && y < image[0].length;

// One thing to fix: an infinite loop will happen if origColor == newColor
```

2. Pay attention to Details

- upon visiting a visited node (coord), must ensure exit condition
- most intuitive way is to record on a separate 2D array visited node

```java
 // OUT: out of index
if (!inArea(image, x, y)) return;
// CLASH: meet other colors, beyond the area of origColor
if (image[x][y] != origColor) return;
// VISITED: don't visit a coordinate twice
if (visited[x][y]) return;
visited[x][y] = true;
image[x][y] = newColor;
```

- This is a common trick to handle graph related - but here a better way is backtracking

```java
void fill(int[][] image, int x, int y,
        int origColor, int newColor) {
    // OUT: out of index
    if (!inArea(image, x, y)) return;
    // CLASH: meet other colors, beyond the area of origColor
    if (image[x][y] != origColor) return;
    // VISITED: visited origColor
    if (image[x][y] == -1) return;

    // choose: mark a flag as visited
    image[x][y] = -1;
    fill(image, x, y + 1, origColor, newColor);
    fill(image, x, y - 1, origColor, newColor);
    fill(image, x - 1, y, origColor, newColor);
    fill(image, x + 1, y, origColor, newColor);
    // unchoose: replace the mark with newColor
    image[x][y] = newColor;
}

// -1 here is special enough to diff from [0, 65535] color values
```

3. Extension: Magic Wand Tool and Minesweeper

- though the background color is blue, cannot ensure all blue pixels are exactly the same (handle minor variation in pixels)
- FloodFill is to fill regions, while Magic Wand Tool is more about filling the edges
- the variation can be handled by `threshold` range to act as `origColor`
- define the second problem: do not color all `origColor` in the region; only care about the edges
- How to find out the nodes at the edges?
- Picturing it, edge nodes have at least one direction that is not `origColor`

```java
int fill(int[][] image, int x, int y,
    int origColor, int newColor) {
    // OUT: out of index
    if (!inArea(image, x, y)) return 0;
    // VISITED: visited origColor
    if (visited[x][y]) return 1;
    // CLASH: meet other colors, beyond the area of origColor
    if (image[x][y] != origColor) return 0;

    visited[x][y] = true;

    int surround = 
          fill(image, x - 1, y, origColor, newColor)
        + fill(image, x + 1, y, origColor, newColor)
        + fill(image, x, y - 1, origColor, newColor)
        + fill(image, x, y + 1, origColor, newColor);

    if (surround < 4)
        image[x][y] = newColor;

    return 1;
}

// all inner nodes will have `surrond` == 4 after traversing four directions;
// all edge nodes will be either OUT or CLASH, resulting `surrond` < 4
// Note the control flow covers all possible scenarios of node (x, y) and the
// value of `surrond` is sum of return values of the 4 recursions - each will fall
// into one of the scenarios

// Pay attention to 2 details in this algorithm: 
// 1. We must use visited to record traversed coordinates instead of backtracking algorithm.
// 2. The order of the if clauses can't be modified. (Why?)

// Minesweeper has extra requirement: 8-directional and returning surrond mine number
```

### Interval Scheduling: Interval Merging

In the Interval Scheduling: Greedy Algo - finding max subset without overlap

Other related problems to interval LeetCode 56 Merge Intervals: given collection of intervals, merge all overlapping ones

```
Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]

The general idea is to observe regular patterns after sorting

- `[start, end]` the last article states the sorting need to be done by `end`; but merging can sort either
- for merging result x, x.start must have the smallest start in intersected intervals, and e.end
must have the largest end too.
- since ordered, x.start is easy, and x.end is analoguous to searching max number

int max_elem = arr[0];
for (int i = 1; i < arr.length; i++)
  max_elem = max(max_elem, arr[i]);
```

```python
def merge(intervals):
  if not intervals: return []
  intervals.sort(key=lambda intv: intv[0])
  res = []
  res.append(intervals[0])

  for i in range(1, len(intervals)):
    curr = intervas[i]
    # quote of last element in res
    last = res[-1]
    if curr[0] <= last[1]:
      last[1] = max(last[1], curr[1])
    else:
      res.append(curr)
  return res
```

### Interval Scheduling: Intersections of Intervals

LeetCode 986 Interval List Intersections - given two lists of closed intervals, each list is pairwise disjoint and sorted. Return the Intersection of the two.

```
Input: 
  A = [[0,2],[5,10],[13,23],[24,25]]
  B = [[1,5],[8,12],[15,24],[25,26]]
Output: 
  [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]

Reminder: The inputs and the desired output are lists of Interval objects, and not arrays or lists.

Note:
  0 <= A.length < 1000
  0 <= B.length < 1000
  0 <= A[i].start, A[i].end, B[i].start, B[i].end < 10^9
```

- Idea is to sort first, then use two pointer

```python
# A, B like [[0,2],[5,10]...]
def intervalIntersection(A, B):
    i, j = 0, 0
    res = []
    while i < len(A) and j < len(B):
        # ...
        j += 1
        i += 1
    return res

# then analyse all cases
# first, for two intervals A, B with a1,a2, b1,b2

# if b2 < a1 or a2 < b1:
#   [a1, a2] and [b1, b2] don't exist intersection
# then what conditions should be met when two intervals has intersection?
# the negative proposition of the above logic

# inverse direction of sign of inequality
# if b2 >= a1 and a2 > b1:
#     [a1, b2] = [b1, b2] has
# then enumerate all cases 
# observe that intersection of intervals has pattern
# if [c1, c2] is intersection, c1 = max(a1, b1), c2 = min(a2, b2)
# while i < len(A) and j < len(B):
#   a1, a2 = A[i][0], A[i][1]
#   b1, b2 = B[j][0], B[j][1]
#   if b2 >= a1 and a2 >= b1:
#     res.append([max(a1, b1), min(a2, b2)])
# where do i and j advance to?
# it depends only on the relationship between a2 and b2
# while i < len(A) and j < len(B):
#   # ...
#   if b2 < b2:
#     j += 1
#   else:
#     i += 1
```

```python
def intervalIntersection(A, B):
  i, j = 0, 0
  res = []
  while i < len(A) and j < len(B):
    a1, a2 = A[i][0], A[i][1]
    b1, b2 = B[j][0], B[j][1]
    # two intervals have intersection
    if b2 >= a1 and a2 >= b1:
      res.append([max(a1, b1), min(a2, b2)])
    if b2 < a2: j += 1
    else:       i += 1
  return res
```

### String Multiplication

When number becomes big, the default data types might overflow (int32); string repr number and perform multiplication.

```
Key is the find pattern (induction) of multiplication process

        1 2 3
          4 5
  ------------
          1 5
        1 0
      0 5
        1 2
      0 8
    0 4
res \ 0 7 3 5
idx 0 1 2 3 4

The product of num1[i] and num2[j] corresponds to res[i+j] and res[i+j+1]
  3 - j = 0
  4 - i = 1
  res = 3 * 4 = 12 with index at res[] 1 (0+1) and 2 (0+1+1)
```

```java
string multiply(string num1, string num2) {
    int m = num1.size(), n = num2.size();
    // the max number of digits in result is m + n
    vector<int> res(m + n, 0);
    // multiply from the rightmost digit
    for (int i = m - 1; i >= 0; i--)
        for (int j = n - 1; j >= 0; j--) {
            int mul = (num1[i]-'0') * (num2[j]-'0');
            // the corresponding index of product in res
            int p1 = i + j, p2 = i + j + 1;
            // add to res
            int sum = mul + res[p2];
            res[p2] = sum % 10;
            res[p1] += sum / 10;
        }
    // the result may have prefix of 0 (which is unused)
    int i = 0;
    while (i < res.size() && res[i] == 0)
        i++;
    // transform the result into string
    string str;
    for (; i < res.size(); i++)
        str.push_back('0' + res[i]);

    return str.size() == 0 ? "0" : str;
}
```

### Pancake Sorting Algo

Given n pieces of pancakes of varied sizes, how to turn it several times with a spatula to make them in order (small up, big down)?

- how to use algo to get a sequence of flips to make cake pile order?
- abstract this problem and use array to repr pancakes heap; solution similar to recusive reverse linked list

1. Analysis of Problem

- why recursive? `void sort(int[] cakes, int n)` - if find largest of first n, then try to flip this pancake to the bottom; then reduce to recursively call sort
- first, find largest of n pancakes
- move it to bottom
- recursively call sort(A, n-1)
- base is n == 1, no need to flip
- e.g. 3rd pancake is the largest, to move it to end: 1) turn first 3 so that largest turns to top; 2) flip all first n so that the largest turns to n-th, which is the last one

```java
// record the reverse operation sequence
LinkedList<Integer> res = new LinkedList<>();

List<Integer> pancakeSort(int[] cakes) {
    sort(cakes, cakes.length);
    return res;
}

void sort(int[] cakes, int n) {
    // base case
    if (n == 1) return;

    // find the index of the largest pancake
    int maxCake = 0;
    int maxCakeIndex = 0;
    for (int i = 0; i < n; i++)
        if (cakes[i] > maxCake) {
            maxCakeIndex = i;
            maxCake = cakes[i];
        }

    // first flip, turn the largest pancake to the top
    reverse(cakes, 0, maxCakeIndex);
    res.add(maxCakeIndex + 1);
    // second flip, turn the largest pancake to the bottom
    reverse(cakes, 0, n - 1);
    res.add(n);

    // recursive
    sort(cakes, n - 1);
}

void reverse(int[] arr, int i, int j) {
    while (i < j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
        i++; j--;
    }
}

// time O(N^2) as n recursion, each a loop;
// Caveat: the length of ops 2(n-1), as each recursion needs 2 flips; there are always n
// layers of recursion, but since base returns result directly, the length is 2(n-1)
// Note this is not optimal (shortest); e.g. [3,2,4,1], flip sequence result
// [3,4,2,3,1,2], but shortest is [2,3,4]
```

### Sliding Window

Double-pointer - left and right of a window. e.g. substring matching problem

At least 9 problems in LeetCode that can be solved using this method.

1. Minimum Window Substring

Given string S and T, find min window in S containing all chars in T in time O(N)

```
S = "ADOBECODEBANC", T = "ABC"

Output: "BANC"
```

```java
// naive
for (int i = 0; i < s.size(); i++)
  for (int j = i + 1; j < s.size(); j++)
    if s[i:j] contains all letters of t:
      update answer
```

Sliding window algo

1. start with double pointers left and right pointing to 0 of S
2. right pointer expand the window until a desirable window that contains all the chars in T
3. move left pointer ahead one by one; if window still desirable keep updating the min window size
4. if window not desirable, repeat step 2 onwards

```
S: E B B A N C F
  left = 0
  right = 0

T: A B C

needs = {A: 1, B: 1, C: 1}
window = {A: 0, B: 0, C: 0}

1. Move right until all chars found
  left = 0
  right = 5
  needs = {A: 1, B: 1, C: 1}
  window = {A: 1, B: 2, C: 1}

2. Move left as long as window still valid
  left = 2
  right = 5
  window = {A: 1, B: 1, C: 1}
```

```java
// pseudocode
string s, t;
int left = 0, right =0;
string res = s;

while (right < s.size())
  window.add(s[right])
  right++
  while (found a valid window)
    res = minLen(res, window)
    window.remove(s[left])
    left++

// How to how 'found a valid window'
// two hash tables as counters; use `needs` to store (char, count) for chars in T,
// `window` to store chars count of T to be found in S;
// if `window` contains all the keys in `needs`, and value of them is greater, then valid

unordered_map<char, int> window;
unordered_map<char, int> needs;
for (char c : t) needs[c]++;

// num of chars valid
int match = 0;

while (right < s.size())
  char c1 = s[right]
  if (needs.count(c1)) // contains key 'c1'
    window[c1]++ // add to window
    if (window[c1] == needs[c1])
      // number of occurrences c1 valid
      match++
  right++

  // when found a valid window
  while (match == needs.size())
    // update res here if found min
    res = minLen(res, window);
    // increase left pointer to make it invalid/valid again
    char c2 = s[left]
    if (needs.count(c2))
      window[c2]-- // remove from window
      if (window[c2] < needs[c2])
        // num of occurrences of c2 no longer valid
        match--;
      left++;
```

```c
// code
string minWindow(string s, string t) {
    // Records the starting position and length of the shortest substring
    int start = 0, minLen = INT_MAX;
    int left = 0, right = 0;

    unordered_map<char, int> window;
    unordered_map<char, int> needs;
    for (char c : t) needs[c]++;

    int match = 0;

    while (right < s.size()) {
        char c1 = s[right];
        if (needs.count(c1)) {
            window[c1]++;
            if (window[c1] == needs[c1]) 
                match++;
        }
        right++;

        while (match == needs.size()) {
            if (right - left < minLen) {
                // Updates the position and length of the smallest string
                start = left;
                minLen = right - left;
            }
            char c2 = s[left];
            if (needs.count(c2)) {
                window[c2]--;
                if (window[c2] < needs[c2])
                    match--;
            }
            left++;
        }
    }
    return minLen == INT_MAX ?
                "" : s.substr(start, minLen);
}

// time O(|S| + |T|) where |S| and |T| repr lengths of S and T.
// Worse cae loop S twice, left and right each once.
// Nested loop N^2? while loop is the total distance that double
// pointer travelled, which is at most 2 meters
```

2. Find All Anagrams in String

Given S and non-empty P find all start indices of P's anagrams in S. - lowercase English letters only with max length <= 20,100; orderless.

```
S: "cbaebabacd" P: "abc"

[0, 6]

Substring with start index = 0 is "cba", which is an anagram of "abc"
Substring with start index = 6 is "bac", ditto.
```

```c
vector<int> findAnagrams(string s, string t) {
    // Init a collection to save the result
    vector<int> res;
    int left = 0, right = 0;
    // Create a map to save the Characters of the target substring.
    unordered_map<char, int> needs;
    unordered_map<char, int> window;
    for (char c : t) needs[c]++;
    // Maintain a counter to check whether match the target string.
    int match = 0;

    while (right < s.size()) {
        char c1 = s[right];
        if (needs.count(c1)) {
            window[c1]++;
            if (window[c1] == needs[c1])
                match++;
        }
        right++;

        while (match == needs.size()) {
            // Update the result if find a target
            if (right - left == t.size()) {
                res.push_back(left);
            }
            char c2 = s[left];
            if (needs.count(c2)) {
                window[c2]--;
                if (window[c2] < needs[c2])
                    match--;
            }
            left++;
        }
    }
    return res;
}

// diff in finding substring of same length
```

3. Longest Substring Without Repeating Characters

Given a string, find length of longest sustring of unique chars.

```
"abcabcbb"

3

Note: answer must be substring (continuous) not subsequence
```

```c
int lengthOfLongestSubstring(string s) {
    int left = 0, right = 0;
    unordered_map<char, int> window;
    int res = 0; // Record maximum length

    while (right < s.size()) {
        char c1 = s[right];
        window[c1]++;
        right++;
        // If a duplicate character appears in the window
        // Move the left pointer
        while (window[c1] > 1) {
            char c2 = s[left];
            window[c2]--;
            left++;
        }
        res = max(res, right - left);
    }
    return res;
}

// when encountering substring problems, first think sliding window technique

// Note when finding max substring, need to update max after the inner while loop
// to ensure validity - par contre, to find min, update inside inner while loop

// Key idea
int left = 0, right = 0;

while (right < s.size())
  window.add(s[right])
  right++;

  while (valid)
    window.remove(s[left])
    left++;

// data type varies to context, such as hash table as counter, or array
// `valid` condition is the trick, might need some code to get updating in real time
// e.g. the first two problems, it seems that solution is so long, but idea still simple
```

### Useful Bit Manipulation

`AND, OR, XOR` most tricks are too obscure, no need to dive deep.

1. Interesting Bit Ops

```java
// OR | and space bar to convert English chars to lowercase
('a' | ' ') = 'a'
('A' | ' ') = 'a'

// AND & and underline converts English to upppercase
('B' & '_') = 'B'

// XOR and space toggle case
('d' ^ ' ') = 'D'
('D' ^ ' ') = 'd'

// Note: due to ASCII encoding - char is number

// Determine if the sign of two numbers are diff
int x = -1, y = 2;
bool f = ((x ^ y) < 0); // true

int x = 3, y = 2;
bool f = ((x ^ y) < 0); // false

// Note: very practical, uses sign bit complement encoding - otherwise need if/else
// Caution on using on multiplication/quotient

// Swap numbers
int a = 1, b = 2;
a ^= b;
b ^= a;
a ^= b;
// a = 2, b = 1

// Plus one
int n = 1;
n = -~n;

// Minus one
int n = 2;
n = ~-n;

// Note: no practical use

// If num is exponent of 2 - if true, its binary must contain only one 1
2^0 = 1 = 0b0001
2^1 = 2 = 0b0010
2^2 = 4 = 0b0100

bool isPowerOfTwo(int n)
  if (n <= 0) return false
  return (n & (n - 1)) == 0
```

2. Algo `n&(n-1)`

```
This eliminates the number n of binary repr of last 1

 rest of number  (least significant 1) (some 0s)
      \         /
   ____\___   /
  |        | |
n  ... 1 1 0 1 0 0

n - 1 : rest remains the same, least significant 1 becomes 0, trailing 0s become 1

n&(n-1) : rest remains the same, all others becomes 0

Use: Count Hamming Weight

Input: 00000000000000000000001011
Output: 3

A binary string has 3 1s

Return several ones in binary of N's one; because n&(n-1) eliminates last one, can use loop
to eliminate 1 and count at the same time until n becomes 0

int hammingWeight(uint32_t n)
  int res = 0;
  while (n != 0)
    n = n & (n - 1)
    res++
  return res
```

### Russian Doll Envelopes Problem

- Trick of sorting lies in clever sorting for preprocessing, transforming the problem and lay foundation for subsequent ops.
- Russian doll envelopes needs to be sorted per ruelss, then covnert into LIS problem;

```
Given a number of envelopes with w,h as (w, h) pairs. 
One envelop can fit into another iif both w,h are greater than the other.
What's the max number of envelopes?
(Rotation disallowed)


Input: [[5,4],[6,4],[6,7],[2,3]]
Output: 3 
Explanation: The maximum number of envelopes you can Russian doll is 3 ([2,3] => [5,4] => [6,7]).
```

- This is a variant of LIS as each legal nesting is a large set of small, equivalent to finding a longest increasing subsequence, and its length is the max number of envelopes that can be nested
- Trick is the standard LIS can only find the longest subsequence in array, here it is a 2D array
- Might compute the area w x h, then LIS on it; but wrong, 1 x 10 is greater than 3 x 3 but invalid
- Solution: first sort w in ascending; if see the same w, sort in descending of h; then use all h as array, compute length of LIS on it returns the answer
- Key is that for pair of same w, the h is sorted in descending; since two envelopes of same w cannot contain each other, reverse ordering ensure that at most one of the pairs of the same w is selected

```java
// envelopes = [[w, h], [w, h]...]
public int maxEnvelopes(int[][] envelopes) {
    int n = envelopes.length;
    // sort by ascending width, and sort by descending height if the width are the same
    Arrays.sort(envelopes, new Comparator<int[]>() 
    {
        public int compare(int[] a, int[] b) {
            return a[0] == b[0] ? 
                b[1] - a[1] : a[0] - b[0];
        }
    });
    // find LIS on the height array
    int[] height = new int[n];
    for (int i = 0; i < n; i++)
        height[i] = envelopes[i][1];

    return lengthOfLIS(height);
}

// The search for LIS can be DP or binary
/* returns the length of LIS in nums */
public int lengthOfLIS(int[] nums) {
    int piles = 0, n = nums.length;
    int[] top = new int[n];
    for (int i = 0; i < n; i++) {
        // playing card to process
        int poker = nums[i];
        int left = 0, right = piles;
        // position to insert for binary search
        while (left < right) {
            int mid = (left + right) / 2;
            if (top[mid] >= poker)
                right = mid;
            else
                left = mid + 1;
        }
        if (left == piles) piles++;
        // put this playing cart on top of the pile
        top[left] = poker;
    }
    // the number of cards is the LIS length
    return piles;
}

// Time O(N log N) as sorting and computing LIS each takes O(N log N)
// Space O(N) as `top` array is needed in LIS
```

- This is hard level and sorting is key. The problem is first transformed into a standard LIS after corret sorting.
- It can be extended to 3D; e.g. nest boxes?
- Might think find nesting in first two dimensions (l, w) and then find LIS in h of the sequence;
- Wrong, this is partial order problem - ascending to 3D will greatly increase difficulty; an advanced data structure Binary Index Tree is needed

### Recursion in Detail

- What's the ins and outs among recursion, divide-and-conquer, DP and greedy?
- Greedy is a subset of DP, while recursion is a thinking for DP and divide-and-conquer.
- Merge Sort is a divide-and-conquer algo: keep dividing unsorted array into smaller sub-problems. Obviously, the sub-problems decomposed by the ranking problem are non-repeating - if some are duplicated (overlapping sub-problems) then DP is used

Recursion

- Think about to transform the problem into sub-problems, rather than how the sub-problem is solved
- Recursion is dividing vertically (enumeration is horizontally) and solves sub-problems in hierarchy
- How to sort a bunch of numbers? Divided into two halves, first align the left half, then right, and finally merge; as for how to arrange the left and right half, read this sentence again...
- How many hairs does Monkey Kind have? One plus the rest.
- How old are you this year? One year plus my age of last year, base is birth year.
- Two key properties: end conditions (answer to simplest sub-problem) and self-invocation (aimed at solving sub-problems)

```java
int func(How old are you this year)
  // simplest sub-problem, end condition
  if (this year == 1999) return my age 0;
  // self-calling to decompose
  return func(How old are you last year) + 1
```

- What is the most successful application of recursion? Mathematical induction - cannot figure out a summation formula, but with a small numbers which seemed containing a kinda law, then compile a formula
- Validity requires induction - assuming that the formula we compiled is true at the k-th, furthermore if it is proved correct at the k+1-th, then the formula is verified correct
- Space is key limit of recursion - stack overflow

```java
void sort(Comparable[] a, int lo, int hi)
  if (lo >= hi) return;
  int mid = lo + (hi - lo) / 2;
  sort(a, lo, mid); // left
  sort(a, mid + 1, hi); // right
  merge(a, lo, mid, hi); // merge the two sides

// Inefficient LinkedList header length calculation

// typical recursive traversal needs extra space O(1)
public int size(Node head)
  int size = 0;
  for (Node p = head; p != null; p = p.next) size++;
  return size;

// space O(N)
public int size(Node head)
  if (head == null) return 0;
  return size(head.next) + 1;
```

- Key: understand what a function does and believe it can solve the task - don't jump into detail. Human brain has tiny stack size.

```java
// Traversing binary tree
void traverse(TreeNode* root)
  if (root == nullptr) return;
  traverse(root->left);
  traverse(root->right);

// Above enough to wipe out any binary tree; the key is to believe traverse(root)

// N-fork tree? Too simple, the same.
void traverse(TreeNode* root)
  if (root == nullptr) return;
  for (child : root->children)
    traverse(child);

// As for pre-order, mid-order, post-order traversal, they are all obvious. N-fork tree
// Obviously no in-order.
// Given a binary tree and a target value, the values in every node is positive or negative,
// return the number of paths in the tree that are equal to the target value, let you write
// the pathSum()

// PathSum III
// root = [10,5,-3,3,2,null,11,3,-2,null,1]
// sum = 8
/*
      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

Return 3. The paths that sum to 8:

1. 5 -> 3
2. 5 -> 2 -> 1
3. -3 -> 11
*/
int pathSum(TreeNode root, int sum)
  if (root == null) return 0;
  return count(root, sum) +
    pathSum(root.left, sum) + pathSum(root.right, sum);

int count(TreeNode node, int sum)
  if (node == null) return 0;
  return (node.val == sum) +
    count(node.left, sum - node.val) + count(node.right, sum - node.val);

// First, it's clear to solve this need full tree traversal
// So traversal framework must appear in main function.
// Each node should see how many eligible paths they and their children have under
// pathSum(): give it a node and a target value, it returns total number of paths in tree rooted at this
// node and the target value
// count(): give it a node and target value, returns a tree rooted at this node and can make up the total
// number of paths strating with the node and target
/* With above tips, comment out the code in detail */
int pathSum(TreeNode root, int sum) {
    if (root == null) return 0;
    int pathImLeading = count(root, sum); // Number of paths beginning with itself
    int leftPathSum = pathSum(root.left, sum); // The total number of paths on the left (Believe he can figure it out)
    int rightPathSum = pathSum(root.right, sum); // The total number of paths on the right (Believe he can figure it out)
    return leftPathSum + rightPathSum + pathImLeading;
}
int count(TreeNode node, int sum) {
    if (node == null) return 0;
    // Can I stand on my own as a separate path?
    int isMe = (node.val == sum) ? 1 : 0;
    // Left brother, how many sum-node.val can you put together?
    int leftBrother = count(node.left, sum - node.val); 
    // Right brother, how many sum-node.val can you put together?
    int rightBrother = count(node.right, sum - node.val);
    return  isMe + leftBrother + rightBrother; // all count i can make up
}
```

Divide and Conquer

- decompose -> solve -> merge
- decompose into sub-problems with the same structure
- perform a recursive solution
- combine the solutions of sub-problems into final

```java
// merge_sort() must clarify the responsibility of the function, sort an incoming array
// Trust it can sort an array is just the same to sorting the two halves of the array separately
void merge_sort(an array) {
    if (some tiny array easy to solve) return;
    merge_sort(left half array);
    merge_sort(right half array);
    merge(left half array, right half array);
}
```

- Because the routine of divide-and-conquer is **decompose -> solve (buttom) -> merge (backtracking)**
- First left and right decomposition, and then processing merge, backtracking is popping stack, equivalent to post-order traversal; `merger` refer to merging two ordered linked-list, they are exactly the same

```java
// Algo 4 - not only thinking important but also coding
public class Merge
  // Do not construct new arrays in the merge(), because it will be called multiple times
  private static Comparable[] aux;

  public static void sort(Comparable[] a)
    aux = new Comparable[a.length];
    sort(a, 0, a.length - 1);

  private static void sort(Comparable[] a, int lo, int hi)
    if (lo >= hi) return;
    int mid = lo + (hi - lo) / 2;
    sort(a, lo, mid);
    sort(a, mid + 1, hi);
    merge(a, lo, mid, hi);

  private static void merge(Comparable[] a, int lo, int mid, int hi)
    int i = lo, j = mid + 1;
    for (int k = lo; k <= hi; k++)
      aux[k] = a[k];
    for (int k = lo; k <= hi; k++)
      if      (i > mid)              { a[k] = aux[j++]; }
      else if (j > hi)               { a[k] = aux[i++]; }
      else if (less(aux[j], aux[i])) { a[k] = aux[j++]; }
      else                           { a[k] = aux[i++]; }

  private static boolean less(Comparable v, Comparable w)
    return v.compareTo(w) < 0;
```

### Backtracking to Subset/Permutation/Combination

- Subset, permutation, combination - these can be solved by backtracking

```
Subset
------

vector<vector<int>> subsets(vector<int>& nums);

nums = [1,2,3], output 8 subsets, including empty set and the set itself, orderless

[], [1], [2], [3], [1,3], [2,3], [1,2], [1,2,3]

The first solution is using idea of maths-induction: suppose knowing results of smaller sub-problem,
then how can I derive the results of the current problem?

Now to find subset [1,2,3], if you have already known subset of [1,2], can you derive subset of
[1,2,3]?

[1,2] -> [], [1], [2], [1,2]

Observe: subset([1,2,3]) - subset([1,2]) = [3], [1,3], [2,3], [1,2,3]

This is to add 3 to each set in the result of subset([1,2])

Or if A = subset([1,2]), then subset([1,2,3]) = A + [ A[i].add(3) for i in 1..len(A) ]

This is typical recursive structure: subset of [1,2,3] can be derived by [1,2], and the subset
of [1,2] can be derived by [1]; base case is when input is empty set - output empty set too
```

```java
vector<vector<int>> subsets(vector<int>& nums) {
    // base case, return an empty set
    if (nums.empty()) return {{}};
    // take the last element
    int n = nums.back();
    nums.pop_back();
    // recursively calculate all subsets of the previous elements
    vector<vector<int>> res = subsets(nums);

    int size = res.size();
    for (int i = 0; i < size; i++) {
        // then append to the previous result
        res.push_back(res[i]);
        res.back().push_back(n);
    }
    return res;
}
```

- It is easy to mistake in calculating the time complexity - find recursion depth and multiply it by hte number of iterations in each recursion; BUT here, depth N, iterations of for loop in each recursion depends on the length of res, which is dynamic
- `res` should be doubled every recursion; so totally `2^N` iterations - or how many subsets of a set size N? At least that number must be added to `res`
- So time `O(2^N)`? No, `2^N` subsets are added to `res` by `push_back`:

```java
for (int i = 0; i < size; i++)
  res.push_back(res[i]); // O(N)
  res.back().push_back(n); // O(1)
```

- Because `res[i]` also an array, `push_back` copies `res[i]` and adds it to the end of the array, so time of one operation is `O(N)`
- Total time `O(N*2^N)`
- If the space used to store the returned results is not calculated, only O(N) recursive stack space required; if calculate the space for `res`, it should be `O(N*2^N)`

```python
# Backtracking 
result = []
def backtrack(Path, Seletion List):
    if meet the End Conditon:
        result.add(Path)
        return

    for seletion in Seletion List:
        select
        backtrack(Path, Seletion List)
        deselect
```

```java
vector<vector<int>> res;

vector<vector<int>> subsets(vector<int>& nums) {
    // record the path
    vector<int> track;
    backtrack(nums, 0, track);
    return res;
}

void backtrack(vector<int>& nums, int start, vector<int>& track) {
    res.push_back(track);
    for (int i = start; i < nums.size(); i++) {
        // select
        track.push_back(nums[i]);
        // backtrack
        backtrack(nums, i + 1, track);
        // deselect
        track.pop_back();
    }
}
// it can be seen that update position of res is in the pre-order traversal, which means
// res is all nodes on the tree
```

Combination

```
n, k -> all combinations of k numbers in [1..n]

n = 4, k = 2 -> orderless, no duplicate [1,2], [1,3], [1,4], [2,3], [2,4], [3,4]

K limits the height of tree, N limits the width of tree

                              []
          /               /          \     \
        [1]              [2]        [3]    [4]
  /      |      \       /   \        |
[1,2]  [1,3]  [1,4]  [2,3]  [2,4]  [3,4]
```

```java
vector<vector<int>>res;

vector<vector<int>> combine(int n, int k) {
    if (k <= 0 || n <= 0) return res;
    vector<int> track;
    backtrack(n, k, 1, track);
    return res;
}

void backtrack(int n, int k, int start, vector<int>& track) {
    // reach the bottom of tree
    if (k == track.size()) {
        res.push_back(track);
        return;
    }
    // note: i is incremented from start 
    for (int i = start; i <= n; i++) {
        // select
        track.push_back(i);
        backtrack(n, k, i + 1, track);
        // deselect
        track.pop_back();
    }
}
// similar to subset, bar the time to update `res` is reaching the bottom of tree
```

3. Permutation

```
nums - does not contain duplicate numbers and return all permutation of them

[1,2,3] -> [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]
```

```java
List<List<Integer>> res = new LinkedList<>();

/* main function, input a uique set of numbers and return all permutations of them */
List<List<Integer>> permute(int[] nums) {
    // record "path"
    LinkedList<Integer> track = new LinkedList<>();
    backtrack(nums, track);
    return res;
}

void backtrack(int[] nums, LinkedList<Integer> track) {
    // trigger the ending condition
    if (track.size() == nums.length) {
        res.add(new LinkedList(track));
        return;
    }

    for (int i = 0; i < nums.length; i++) {
        // exclud illegal selections
        if (track.contains(nums[i]))
            continue;
        // select
        track.add(nums[i]);
        // go to the next decision tree
        backtrack(nums, track);
        // deselect
        track.removeLast();
    }
}
// Tree of permutation is relatively symmetrical, and the tree of the combination problem has fewer right nodes
// `contains` to exclude the numbers that have been selected in `track` each time; while the combination
// problem passes a `start` parameter to exclude the numbers before `start` index

/*
The subset problem can use the idea of mathematical induction: 
assuming that the results of a smaller problem are known,
and thinking about how to derive the results of the original problem.
You can also use the backtracking algorithm, using the start parameter to exclude selected numbers.

The combination problem uses the backtracking idea, and the results can be expressed 
as a tree structure. We only need to apply the backtracking algorithm template.
The key point is to use a start to exclude the selected numbers.
The permutation problem uses the backtracking idea, and it can also be expressed
as a tree structure to apply the algorithm template. The key point is to use the
contains method to exclude the selected numbers. There is detailed analysis previously.
Here we mainly compare it with the combination problem.

Keeping the shape of these trees in mind is enough to deal with most backtracking
algorithm problems. It is nothing more than the pruning of start or contains. There is no other trick.
*/
```

### Shuffle Algo

1. Shuffle Algo - exchange randomly selected elements

```java
// A random integer in the closed interval [min, Max] is obtained
int randInt(int min, int max);

// First case
void shuffle(int[] arr) {
    int n = arr.length();
    /*** The only difference is these two lines ***/
    for (int i = 0 ; i < n; i++) {
        // Randomly select an element from i to the last
        int rand = randInt(i, n - 1);
    /*********************************************/
        swap(arr[i], arr[rand]);
    }
}

// Second case
    for (int i = 0 ; i < n - 1; i++)
        int rand = randInt(i, n - 1);

// Third cse
    for (int i = n - 1 ; i >= 0; i--)
        int rand = randInt(0, i);

// Forth case
    for (int i = n - 1 ; i > 0; i--)
        int rand = randInt(0, i);
```

- To analyze the correctness: result must have `n!` possibilities. - because array of length n has full permutation of `n!` (i.e. total number of disruption results are n!)

```java
// Suppose an arr is passed in like this
int[] arr = {1,3,5,7,9};

void shuffle(int[] arr) {
    int n = arr.length(); // 5
    for (int i = 0 ; i < n; i++) {
        int rand = randInt(i, n - 1);
        swap(arr[i], arr[rand]);
    }
}
```

- At first iteration of for loop, `i=0`, range of `rand` is `[0,4]` and there are 5 possible values

![first-iteration](https://1829267701-files.gitbook.io/~/files/v0/b/gitbook-legacy-files/o/assets%2F-M1hB-LnPpOmZGsmxY7T%2F-M2cPUHf-Qho1pEMDPxb%2F-M2cPVVsGlFyTv7ujiv-%2F1.png?generation=1584447237652384&alt=media)

- On second, `i=1`, range is `[1,4]` - four possible values

![second-iteration](https://1829267701-files.gitbook.io/~/files/v0/b/gitbook-legacy-files/o/assets%2F-M1hB-LnPpOmZGsmxY7T%2F-M2cPUHf-Qho1pEMDPxb%2F-M2cPVVywuox69Fx5RN4%2F2.png?generation=1584447238673631&alt=media)

- and so on, until last `i=4` and range is `[4,4]` with one possible value

![last-iteration](https://1829267701-files.gitbook.io/~/files/v0/b/gitbook-legacy-files/o/assets%2F-M1hB-LnPpOmZGsmxY7T%2F-M2cPUHf-Qho1pEMDPxb%2F-M2cPVW-cpHRT50ZYhBj%2F3.png?generation=1584447234746572&alt=media)

- all possible outcomes is `n! = 5!` so algo is correct
- In second case, the previous iteration the same, with only one iteration missing. 

```java
// Second case
// arr = {1,3,5,7,9}, n = 5
    for (int i = 0 ; i < n - 1; i++)
        int rand = randInt(i, n - 1);
```

- Third way is the first way, just iterating the array from back to front; fourth way is second way from the back

```java
// wrong idea
void shuffle(int[] arr) {
    int n = arr.length();
    for (int i = 0 ; i < n; i++) {
        // Every time, elements are randomly selected from 
        //the closed interval [0, n-1] for exchange 

        int rand = randInt(0, n - 1);
        swap(arr[i], arr[rand]);
    }
}

// because all possible outcomes are n^n not n!
// arr = [1,2,3], 3^3 = 27 cannot be divisible by 6, there must be some cases that are biased, more likely to occur
```

```java
// First idea: enumerate all permutaitons of the array and make histogram
// as verifcation, only need small n
void shuffle(int[] arr);

// Monte Carlo
int N = 1000000;
HashMap count; // As histogram
for (i = 0; i < N; i++) {
    int[] arr = {1,2,3};
    shuffle(arr);
    // At this time, arr has been disrupted 
    count[arr] += 1；
}
for (int feq : count.values()) 
    print(feq / N + " "); // frequency

// Second idea: there is only one 1 in arr, others all 0, so mess up a million times and record 
// occurences of 1 per index, and if number of per index is about the same, hten mess is equal
void shuffle(int[] arr);

// Monte Carlo method
int N = 1000000;    
int[] arr = {1,0,0,0,0};
int[] count = new int[arr.length];
for (int i = 0; i < N; i++) {
    shuffle(arr); // disrupt arr
    for (int j = 0; j < arr.length; j++) 
        if (arr[j] == 1) {
            count[j]++;
            break;
        }
}
for (int feq : count) 
    print(feq / N + " "); // frequency
```
