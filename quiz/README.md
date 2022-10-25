# High Frequency Problem

https://labuladong.gitbook.io/algo-en/iv.-high-frequency-interview-problem/print_primenumbers

## Find Prime Number

- Tricky to write a function to check prime

```java
// naive
int countPrimes(int n)
  int count = 0;
  for (int i = 2; i < n; i++)
    if (isPrime(i)) count++;
  return count;

boolean isPrime(int n)
  for (int i = 2; i < n; i++)
    if (n % i == 0)
      return false;
    return true;

// time O(N^2)

// modify loop
for (int i = 2; i * i <= n; i++)
// i need not traverse to n, but only to sqrt(n)
/*
12 = 2 x 6
12 = 3 x 4
12 = sqrt(12) x sqrt(12)
12 = 4 x 3
12 = 6 x 2
*/
// There's reverse elements at the inflection point
// now time reduced to O(sqrt(N))
```

Efficient Implementation of `countPrimes`

- Reverse thinking - 2 is prime, then 2 x 2 = 4, 3 x 2 = 6, 4 x 2 = 8 ... none is prime.
- 3 is prime. 3 x 2 = 6, 3 x 3 = 9, ... none is prime.

```java
int countPrimes(int n) {
    boolean[] isPrim = new boolean[n];
    // Initialize the arrays to true
    Arrays.fill(isPrim, true);

    for (int i = 2; i < n; i++) 
        if (isPrim[i]) 
            // Multiples of i cannot be prime
            for (int j = 2 * i; j < n; j += i) 
                    isPrim[j] = false;

    int count = 0;
    for (int i = 2; i < n; i++)
        if (isPrim[i]) count++;

    return count;
}

// Two subtle areas to optim
// First, the symmetry reduces loop to [2, sqrt(n)]; similarly, outer loop
for (int i = 2; i * i < n; i++)
  if (isPrime(i))
// Then, it's hard to notice inner loop has redundancy
// n = 25, i = 4, algo mark numbers such as 4 x 2 = 8, 4 x 3 = 12, but these 
// have been marked by 2 x 4, 3 x 4 (i = 2 and i = 3)
// j traverse from square of i instead of starting from 2 * i
for (int j = i * i; j < n; j += 1)
  isPrime[j] = false;

// Sieve of Eratosthenes
```

## Drop Water

- Given n non-negative integers repr an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.

```
[0,1,0,2,1,0,1,3,2,1,2,1] -> 6

'fill in holes'

Brute force -> memo -> two pointer

Final code:
time O(N)
space O(1)
```

Core Idea

- Think not the whole, but part; focus on how to solve unit problem (char instead of string)
- Between `[1,0,1]`, i = 0, why does it hold 2 grids?

```
Because height[i] == 0, height[i] can hold up to 2 grids
Because it depends on both the highest on left and right - l_max, r_max
Thus height at position i is min(l_max, r_max)

water[i] = min(
      max(height[0..i]),
      max(height[i..end]),
  ) - height[i]
```

```java
// brute force
int trap(vector<int>& height) {
    int n = height.size();
    int ans = 0;
    for (int i = 1; i < n - 1; i++) {
        int l_max = 0, r_max = 0;
        // find the highest column on the right
        for (int j = i; j < n; j++)
            r_max = max(r_max, height[j]);
        // find the highest column on the right
        for (int j = i; j >= 0; j--)
            l_max = max(l_max, height[j]);
        // if the position i itself is the highest column
        // l_max == r_max == height[i]
        ans += min(l_max, r_max) - height[i];
    }
    return ans;
}

// Memo Caching
int trap(vector<int>& height) {
    if (height.empty()) return 0;
    int n = height.size();
    int ans = 0;
    // arrays act the memo
    vector<int> l_max(n), r_max(n);
    // initialize base case
    l_max[0] = height[0];
    r_max[n - 1] = height[n - 1];
    // calculate l_max from left to right
    for (int i = 1; i < n; i++)
        l_max[i] = max(height[i], l_max[i - 1]);
    // calculate r_max from right to left
    for (int i = n - 2; i >= 0; i--) 
        r_max[i] = max(height[i], r_max[i + 1]);
    // calculate the final result
    for (int i = 1; i < n - 1; i++) 
        ans += min(l_max[i], r_max[i]) - height[i];
    return ans;
}

// Two pointers
int trap(vector<int>& height) {
    int n = height.size();
    int left = 0, right = n - 1;

    int l_max = height[0];
    int r_max = height[n - 1];

    while (left <= right) {
        l_max = max(l_max, height[left]);
        r_max = max(r_max, height[right]);
        left++; right--;
    }
}

// l_max is highest among height[0..left], etc.
int trap(vector<int>& height) {
    if (height.empty()) return 0;
    int n = height.size();
    int left = 0, right = n - 1;
    int ans = 0;

    int l_max = height[0];
    int r_max = height[n - 1];

    while (left <= right) {
        l_max = max(l_max, height[left]);
        r_max = max(r_max, height[right]);

        // ans += min(l_max, r_max) - height[i]
        if (l_max < r_max) {
            ans += l_max - height[left];
            left++; 
        } else {
            ans += r_max - height[right];
            right--;
        }
    }
    return ans;
}

// In memo version, l_max[i] and r_max[i] repr highest of height[0..i] and height[i..end]
ans += min(l_max[i], r_max[i]) - height[i];
// But two pointers they repr highest of height[0..left] and height[right..end]
if (l_max < r_max) {
    ans += l_max - height[left];
    left++; 
}
// l_max repr highest on left of `left` pointer, but `r_max` not always the highest on right of `left`
// Need to think: focus on min(l_max, r_max) -> l_max < r_max, so it matters not whether r_max is
// highest on right, the key is water capacity in height[i] depends only on l_max
```

## Remove Duplicate From Sorted Sequence

- Given sorted array, remove dup in-place (do not alloc extra space with space O(1))
- since sorted, dups are consecutive, but deletion is O(N^2) in array
- Array general technique: avoid deleting in the middle, then find ways to swap to last
- Two pointers (fast slow)
- `slow` go to the back, `fast` moves ahead; on finding unique element, `slow` move forward; `fast` traverses entire array, `nums[0..slow]` is a unique element, and all subsequent ones are repeated

```java
int removeDuplicates(int[] nums) {
    int n = nums.length;
    if (n == 0) return 0;
    int slow = 0, fast = 1;
    while (fast < n) {
        if (nums[fast] != nums[slow]) {
            slow++;
            // Maintain no repetition of nums[0..slow] 
            nums[slow] = nums[fast];
        }
        fast++;
    }
    //The length is index + 1 
    return slow + 1;
}

// Extending this to sorted list - bar array assignment is now pointer
ListNode deleteDuplicates(ListNode head) {
    if (head == null) return null;
    ListNode slow = head, fast = head.next;
    while (fast != null) {
        if (fast.val != slow.val) {
            // nums[slow] = nums[fast];
            slow.next = fast;
            // slow++;
            slow = slow.next;
        }
        // fast++
        fast = fast.next;
    }
    // The list disconnects from the following repeating elements
    slow.next = null;
    return head;
}
```

## Longest Palindromic Substring

- Double pointer framework
- Reverse S: 'abacd' -> 'dcaba' and longest common string is 'aba'; but 'aacxycaa' -> 'aacyxcaa' -> 'aac' this is wrong ('aa')

```java
// Core idea: start scanner from mid point
for 0 <= i < len(s):
    find a palindrome that set s[i] as its mid point
    update the answer

// when length of string is even, 'abba' above fails
// better version
for 0 <= i < len(s):
    find a palindrome that set s[i] as its mid point
    find a palindrome that set s[i] and s[i + 1] as its mid point
    update the answer
// Note: may encounter outofindex

// Implementation
string palindrome(string& s, int l, int r) {
    // avoid outOfIndex error
    while (l >= 0 && r < s.size()
            && s[l] == s[r]) {
        // scanning toward both directions
        l--; r++;
    }
    // return a palindrome that set s[l] and s[r] as mid point
    return s.substr(l + 1, r - l - 1);
}

// Handle both odd and even length
for 0 <= i < len(s):
    # find a palindrome that set s[i] as its mid 
    palindrome(s, i, i)
    # find a palindrome that set s[i] and s[i + 1] as its mid  
    palindrome(s, i, i + 1)
    update the answer

// Full
string longestPalindrome(string s) {
    string res;
    for (int i = 0; i < s.size(); i++) {
        // find a palindrome that set s[i] as its mid 
        string s1 = palindrome(s, i, i);
        // find a palindrome that set s[i] and s[i + 1] as its mid  
        string s2 = palindrome(s, i, i + 1);
        // res = longest(res, s1, s2)
        res = res.size() > s1.size() ? res : s1;
        res = res.size() > s2.size() ? res : s2;
    }
    return res;
}

// time O(N^2)
// space O(1)
// DP can also solve but need O(N^2) space to store DP table

// Manacher's Algo requires only O(N) time
```

## Reverse Linked List in K Group

Given a linked list, reverse the nodes of a linked list k at a time and return its modified list. k is a positive integer and is <= length. If the number of nodes != multiple of k then left-out nodes should remain as it is. `1-2-3-4-5`, `k = 2 -> 2-1-4-3-5`, `k=3 -> 3-2-1-4-5`

```
- linked list is of recursion and iteration
- reverseKGroup(head, 2) reverse the linked list with 2 nodes as a group

After first sub-problem [0..2] reversed, the remaining nodes also form a linked list
but it's shorter.

newHead   head
     \   / 
   1<-2 3->4->5->6->NULL
    \
     reverseKGroup(head, 2)

reverseKGroup(cur, 2) recursively since it's the same data structure between prime and sub problems

1. Reverse first k nodes

    head   newHead
       \    / 
  NULL<-1<-2 3->4->5->6->NULL

2. Reverse list with k+1 node as head

    newHead   head
          \   / 
  NULL<-1<-2 3->4->5->6->NULL

3. Merge result of above two steps

 newHead  head
      \   / 
    1<-2 3->4->5->6->NULL
     \
    reverseKGroup(head, 2)
```

```java
// reverse the linked list with node a as head
ListNode reverse(ListNode a) {
    ListNode pre, cur, nxt;
    pre = null; cur = a; nxt = a;
    while (cur != null) {
        nxt = cur.next;
        // reverse node one by one
        cur.next = pre;
        // update pointer
        pre = cur;
        cur = nxt;
    }
    // return head node of the reversed linked list
    return pre;
}

// pre cur nxt -> keep moving forward till cur == NULL

// To reverse nodes in interval a..b - change NULL to b
/** reverse the nodes of interval [a, b), which is left-closed and right-open */
ListNode reverse(ListNode a, ListNode b) {
    ListNode pre, cur, nxt;
    pre = null; cur = a; nxt = a;
    // just change the condition of quit
    while (cur != b) {
        nxt = cur.next;
        cur.next = pre;
        pre = cur;
        cur = nxt;
    }
    // return head node of the reversed linked list
    return pre;
}

// Given reversing partial list
ListNode reverseKGroup(ListNode head, int k) {
    if (head == null) return null;
    // interval [a, b) includes k nodes to be reversed
    ListNode a, b;
    a = b = head;
    for (int i = 0; i < k; i++) {
        // base case
        if (b == null) return head;
        b = b.next;
    }
    // reverse first k nodes
    ListNode newHead = reverse(a, b);
    // merge all reversed internals
    a.next = reverseKGroup(b, k);
    return newHead;
}
// Note interval of `reverse` is [a, b)

/*
      a  newHead       b
       \   |          /
 NULL x 1<-2 reverse(3->4->5->...)
        |______^

newHead _____
   \   /     \
 1<-2 3<-4 5<-6
  \______^  \
             null
*/
```

## Check Validity of Parenthesis

Example: editor and compiler check code for parenthesis closure
Given a string containing just `(,),[,],{,}` determine if input is valid

```
First, if only one type of parenthesis. - every ) must have (

bool isValid(string str) {
    // the number of left parenthesis to be matched
    int left = 0;
    for (char c : str) {
        if (c == '(')
            left++;
        else // encounter right parenthesis
            left--;

        if (left < 0)
            return false;
    }
    return left == 0;
}

Multiple separate handle of left1, left2,... seemingly solves multiple parenthesis
But it fails - (()) is valid in the case of one parenthesis, while [(]) is not

Just tracking the pairs is not enough - need more info. STACK

Stack is FILO - use left stack - find recent left in stack and check if matches
```

```java
bool isValid(string str) {
    stack<char> left;
    for (char c : str) {
        if (c == '(' || c == '{' || c == '[')
            left.push(c);
        else // character c is right parenthesis
            if (!left.empty() && leftOf(c) == left.top())
                left.pop();
            else
                // not match with recent left parenthesis
                return false;
    }
    // whether all left parenthesis are matched
    return left.empty();
}

char leftOf(char c) {
    if (c == '}') return '{';
    if (c == ')') return '(';
    return '[';
}
```

## Find Missing Element

Given an array of n from 0, 1, 2, ..., n find out missing number. `[3,0,1] -> 2`

```
Naive time for sorting O(N log N) while HashSet ~ O(N) but space O(N)

Third solution: Bit Operation

XOR has a special property: result of a number XOR itself is 0, and result of a number
with 0 is itself.

XOR satisfy the Exchange Law and Communicative Law

2 ^ 3 ^ 2 = 3 ^ (3 ^ 2) = 3 ^ 0 = 3

idx 0 1 2 3
num 0 3 1 4

Assuming index increments by 1 from [0, n) to [0, n) and let each element to be placed
at the index of its value

idx 0 1 2 3 4
num 0 1   3 4

All elements and their indices will be a pair except missing element
If we can find out index 2 is missing, then that's the missing element

Perform XOR to all elements and their indices - only missing element will be left (all
other 0)
```

```java
int missingNumber(int[] nums) {
    int n = nums.length;
    int res = 0;
    // XOR with the new index first
    res ^= n;
    // XOR with the all elements and the other indices
    for (int i = 0; i < n; i++)
        res ^= i ^ nums[i];
    return res;
}
```

- Now time is O(N) and space O(1)
- Summation of Arithmetic Progression (AP) - given an arithmetic progression with an missing
- Consequently the number is just `sum(0,1,..n) - sum(nums)`

```java
int missingNumber(int[] nums) {
    int n = nums.length;
    // Formula: (head + tail) * n / 2
    int expect = (0 + n) * (n + 1) / 2;

    int sum = 0;
    for (int x : nums) 
        sum += x;
    return expect - sum;
```

- Bug - integer overflow - if product is too big
- To fix, perform subtraction while summing up, similar to bit ops

```
index  0  1  2  3 
value  0  3  1  4

missing = 4^(0^0)^(1^3)^(2^1)^(3^4)
        = (4^4)^(0^0)^(1^1)^(3^3)^2
        = 0^0^0^0^2
        = 2

Subtracting each element from its index, then sum up the diffs, result will be the missing
```

```java
public int missingNumber(int[] nums) {
    int n = nums.length;
    int res = 0;
    // Added index
    res += n - 0;
    // Summing up the differences between the remaining indices and elements
    for (int i = 0; i < n; i++) 
        res += i - nums[i];
    return res;
}
// As both addition and subtraction satisfy the Exchange Law and Communicative Law, we
// can always eliminate paired numbers, left with missing ones.
```

## Pick Elements From Arbitrary Sequence

- LeetCode 382, 398 Reservoir Sampling which is random possibility
- Given a linked list with unknown length, return one node from the linked list with traversing the linked list only once
- The meaning of random is uniform random, each node has 1/n
- Simple idea is to first walk the whole list then get total n; then make an index from random number in range `[1, n]`; finding the corresponding node of the index means finding randomly selected node
- But the requirement is, traversing the linked list only once
- More general, given length-unknown sequence, how to select k elements randomly from it?
- Such kind of problems follows framework of Reservoir Sampling

```java
// First, should try to solve selecting only one element
// The trick of random selection is actually dynamic
// Starting with 5 elements, now add one - may keep selecting a or changing
// to the new element b as the result. But how to know it's fair

// Conclusion: if at i-th element, the possibility of selecting that
// element should be 1/i and the possibility to keep the original choice is 1 - 1/i

/* return the value of a random node from the linked list */
int getRandom(ListNode head) {
    Random r = new Random();
    int i = 0, res = 0;
    ListNode p = head;
    // while iterate through the linked list
    while (p != null) {
        // generate an integer in range [0, i) 
        // the possibility of the integer equals to 0 is 1/i
        if (r.nextInt(++i) == 0) {
            res = p.val;
        }
        p = p.next;
    }
    return res;
}

// As for randomness, the code is usually short, but the key problem is how to
// prove correctness. Why uniform random when updating result with 1/i possibility

// Proof: assume n in total, to make the possibility of selecting each element 1/n
// Then for the i-th, the possibility of selecting:
/*
= 1/i * (1- 1/(i+1)) * (1 - 1/(i+2)) * ... (1 - 1/n)
= 1/i * i/(i+1) * (i+1)/(i+2) * (n-1)/n
= 1/n
*/
```

```
At the i-th position, the possibility of i-th element to be selected is 1/i.
At the i+1-th position, the possibility of i-th element not to be replaced is
1 - 1/(i+1). And similarly, the products of all the possibilities until the n-th
position should be the final possibility of the i-th element is chosen. The result is 1/n.

Similarly, if need to select k randomly. The only thing need to do is to keep the possibility
of selecting i-th at i-th position k/i and make the possibility of keeping the original selection
1 - k/i
```

```java
/* return the values of k random nodes from the linked list */
int[] getRandom(ListNode head, int k) {
    Random r = new Random();
    int[] res = new int[k];
    ListNode p = head;

    // select first k elements by default
    for (int j = 0; j < k && p != null; j++) {
        res[j] = p.val;
        p = p.next;
    }

    int i = k;
    // while iterate the linked list
    while (p != null) {
        // generate an integer in range [0, i) 
        int j = r.nextInt(++i);
        // the possibility of the integer less than k is k/i
        if (j < k) {
            res[j] = p.val;
        }
        p = p.next;
    }
    return res;
}
// Although, every time the possibility of updating the selection increased by k times,
// for the distinct i-th element, the possibility should be multiplied by 1/k,
// which comes back to the last reduction.
```

```
The time is O(N), a more optim is using geometric distribution with time O(k + k log (n/k))
Fisher-Yates shuffle is another, requiring array for random access to elements.

Another is make each element related to a random number, and then insert each into a heap with
capacity k. Sort the heap by the related random number, then the rest k elements are also randomized

This method is not fast but afford sampling with weights which is useful

Q1: How could you carry out weighted random sampling for samples with weight? 
For example, given an array w and every elements w[i] representing the weight.
Can you write an algorithm to select the index with the corresponding weight.
When w = [1,99], you should make the possibility to select index 0 becoome 1%
and the possibility to select index 1 become 99%.

Q2: Implement a generator class, and a very long array would be parsed into the constructor.
Can you implement the randomGet method, which makes sure that every time when called,
it returns one element of the array randomly and it can't return the same element in
multiple callings. Besides, the array could not be modified in any form, and the time
complexity should be O(1).
```

## Binary Search

- Search array with duplicates could result in any of the target (left or right);
- how can binary search be applied when search space is ordered, pruning improves efficiency

```
Koto eating banana
N piles of bananas, i-th pile has piles[i]; the guards have gone and will come back in H hours
Koto can decide her bananas-per-hour eating speed K;
each hour she chooses some pile and eats K from it;
if pile < K, she eats all of them for the hour
Koto wants to finish eating all bananas before guards come back
Return min integer K, such that she can eat all within H hours

[3,6,7,11] H = 8 -> 4

Brute-force - min speed of eating in H hours - `speed`
Min is 1 and max is max(piles)
As long as it starts from 1 and exhausts to max(piles), once it
is found that a certain value can eat all in H hours, this value is the min speed
```

```java
int minEatingSpeed(int[] piles, int H) {
    // the maximum value of piles
    int max = getMax(piles);
    for (int speed = 1; speed < max; speed++) {
        // wherher can finish eating banana in H hours at speed
        if (canFinish(piles, speed, H))
            return speed;
    }
    return max;
}

// Note this for loop is linear search
// Binary search can be used to find left boundary

int minEatingSpeed(int[] piles, int H) {
    // apply the algorithms framework for searching the left boundary
    int left = 1, right = getMax(piles) + 1;
    while (left < right) {
        // prevent overflow
        int mid = left + (right - left) / 2;
        if (canFinish(piles, mid, H)) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
}

// Time complexity O(N)
boolean canFinish(int[] piles, int speed, int H) {
    int time = 0;
    for (int n : piles) {
        time += timeOf(n, speed);
    }
    return time <= H;
}

int timeOf(int n, int speed) {
    return (n / speed) + ((n % speed > 0) ? 1 : 0);
}

int getMax(int[] piles) {
    int max = 0;
    for (int n : piles)
        max = Math.max(n, max);
    return max;
}

// time O(N log N)
```

- Extension: transportation problem; i-th package on the conveyor belt has a weight of `weight[i]`; each day oad ship with packages on the conveyor (in order given by weight) may not load more than max weight capacity
- return least weight capacity of ship that will result in all packages being shipping within D days

```

Input: weights = [1,2,3,4,5,6,7,8,9,10], D = 5
Output: 15
Explanation: 
A ship capacity of 15 is the minimum to ship all the packages in 5 days like this:
1st day: 1, 2, 3, 4, 5
2nd day: 6, 7
3rd day: 8
4th day: 9
5th day: 10

Note that the cargo must be shipped in the order given, so using a ship of capacity 14
and splitting the packages into parts like (2, 3, 4, 5), (1, 6, 7), (8), (9), (10) is not allowed. 
```

- first find min and max of `cap` as `max(weights)` and `sum(weights)`
- require min load, so a binary search on left boundary can be used to optim linear search

```java
// find the left boundary using binary search
int shipWithinDays(int[] weights, int D) {
    // minimum possible load
    int left = getMax(weights);
    // maximum possible load + 1
    int right = getSum(weights) + 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (canFinish(weights, D, mid)) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
}

// If the load is cap, can I ship the goods within D days？
boolean canFinish(int[] w, int D, int cap) {
    int i = 0;
    for (int day = 0; day < D; day++) {
        int maxCap = cap;
        while ((maxCap -= w[i]) >= 0) {
            i++;
            if (i == w.length)
                return true;
        }
    }
    return false;
}

// Through these two examples, 
for (int i = 0; i < n; i++)
    if (isOK(i))
        return ans;
```

## Scheduling Seats

- LC-885, Practice goes deeper than theory
- suppose a room with a row of N seats; their indexes are `[0..n-1]`; candidates will successively enter the room and leave anytime; arrange seats so that whenever a student enters, max the distance between him and the nearest other; if there are more than one such seats, arrange him to the seat with the smallest index

```java
class ExamRoom {
    // constructor, receive the N which means total number of seats  
    public ExamRoom(int N);
    // when a candidate comes, return to the seat assigned for him
    public int seat();
    // The candidate in the position P now left
    // It can be considered that there must be a candidate in the position P
    public void leave(int p);
}

```

- e.g. five seats `[0..4]`; when person 1 enters`seat()` it is valid to sit at lowest index, position 0
- person 2 `seat()` means position 4 (max distance)
- person 3 goes to seat 2
- if another enters, both seat 1 and 3 are valid, take smaller index 1

```
If regard every two adjacent persons as two endpoints of a line segment, the new arrangement is
that, find the longest line segment, let this person 'dichotomy' the line segment in middle, 
then the middle point is the seat assigned; `leave(P)` is to remove the endpoint P so as to
merge two adjacent segments into one

Which data structure to pick?
```

## Union-Find

## Find Subsequence with Binary Search

## One Liner

## Find Dup and Missing

## Check Palindrom Linked List
