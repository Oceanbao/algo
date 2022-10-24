## Data Structure

### Binary Heap and Priority Queue

```java
// the index of the parent node
int parent(int root) {
    return root / 2;
}
// index of left child
int left(int root) {
    return root * 2;
}
// index of right child
int right(int root) {
    return root * 2 + 1;
}
```

```python
class Heap:
  def __init__(self, comparator):
    self.arr = []
    self.comparator = comparator

  def get_parent(self, k):
    return (k - 1) // 2

  def peek(self):
    return self.arr[0]

  def push(self. v):
    self.arr.append(v)
    self._heapify_up(len(self.arr) - 1)

  def pop(self):
    self.arr[0] = self.arr[-1]
    self.arr.pop()
    self._heapify_down(0)

  def _heapify_up(self, k):
    parent = self.get_parent(k)
    if parent == -1:
      return
    if not self.comparator(self.arr[parent], self.arr[k]):
      self.arr[parent], self.arr[k] = self.arr[k], self.arr[parent]
      self._heapify_up(parent)

  def _heapify_down(self, k):
    if k >= len(self.arr):
      return
    left = 2 * k + 1
    right = 2 * k + 2
    index = k
    if left < len(self.arr) and not self.comparator(self.arr[index], self.arr[left]):
      index = left
    if right < len(self.arr) and not self.comparator(self.arr[index], self.arr[right]):
        index = right
    if index != k:
        self.arr[index], self.arr[k] = self.arr[k], self.arr[index]
        self._heapify_down(index)

  def make_heap(self, arr):
    for _, v in enumerate(arr):
      self.arr.append(v)

    index = (len(self.arr) - 1) // 2
    for i in range(index, -1, -1):
      self._heapify_down(i)

min_heap = Heap(lambda a, b : a < b)
max_heap = Heap(lambda a, b : a > b)

arr = [4, 3, 5, 2, 1, 6, 8, 7, 9]
min_heap.make_heap(arr)
max_heap.make_heap(arr)

print(min_heap.peek())
print(max_heap.peek())

min_heap.push(0)
max_heap.push(10)

print(min_heap.peek())
print(max_heap.peek())

min_heap.pop()
max_heap.pop()

print(min_heap.peek())
print(max_heap.peek())
```

### LRU Cache

Just a cache clean-up strategy.

LRU algo is about DS design with given capacity and implement 2 operations. 

`put(key, val)` to store key-value pair
`get(key)` to return value by key or -1 else

- Ordered: required for finding least recently used and longest unused
- Fast Search: find key in cahce
- Fast Delete: if cache full, delete last element
- Fast Insert: insert to head on each visit

Which DS?

Hashtable is fast search but unordered; Linked-list ordered and fast insert-delete but slow search.
Mixing the two we get hash-linked-list.

- Hashtable for fast search to linked-list
- Doubly-linked-list required, see implementation for why

```java
class Node {
    public int key, val;
    public Node next, prev;
    public Node(int k, int v) {
        this.key = k;
        this.val = v;
    }
}

class DoubleList {  
    // Add x at the head, time complexity O(1)
    public void addFirst(Node x);

    // Delete node x in the linked list (x is guaranteed to exist)
    // Given a node in a doubly linked list, time complexity O(1)
    public void remove(Node x);

    // Delete and return the last node in the linked list, time complexity O(1)
    public Node removeLast();

    // Return the length of the linked list, time complexity O(1)
    public int size();
}

// Skipping implement detail of doubly-linked-list
// In order to delete node, not only need to get pointer of node itself, but
// also update the node before and after, thus need DLL for O(1)

// key associated with Node(key, val)
HashMap<Integer, Node> map;
// Node(k1, v1) <-> Node(k2, v2)...
DoubleList cache;

/*
int get(int key) {
    if (key does not exist) {
        return -1;
    } else {        
        bring (key, val) to the head;
        return val;
    }
}

void put(int key, int val) {
    Node x = new Node(key, val);
    if (key exists) {
        delete the old node;
        insert the new node x to the head;
    } else {
        if (cache is full) {
            delete the last node in the linked list;
            delete the associated value in map;
        } 
        insert the new node x to the head;
        associate the new node x with key in map;
    }
}
*/

class LRUCache {
    // key -> Node(key, val)
    private HashMap<Integer, Node> map;
    // Node(k1, v1) <-> Node(k2, v2)...
    private DoubleList cache;
    // Max capacity
    private int cap;

    public LRUCache(int capacity) {
        this.cap = capacity;
        map = new HashMap<>();
        cache = new DoubleList();
    }

    public int get(int key) {
        if (!map.containsKey(key))
            return -1;
        int val = map.get(key).val;
        // Using put method to bring it forward to the head
        put(key, val);
        return val;
    }

    public void put(int key, int val) {
        // Initialize new node x
        Node x = new Node(key, val);

        if (map.containsKey(key)) {
            // Delete the old node, add to the head
            cache.remove(map.get(key));
            cache.addFirst(x);
            // Update the corresponding record in map
            map.put(key, x);
        } else {
            if (cap == cache.size()) {
                // Delete the last node in the linked list
                Node last = cache.removeLast();
                map.remove(last.key);
            }
            // Add to the head
            cache.addFirst(x);
            map.put(key, x);
        }
    }
}

// Beware of the need to store key-value pair in LL instead of value
if (cap == cache.size()) {
    // Delete the last node
    Node last = cache.removeLast();
    map.remove(last.key);
}
// Deleting last node (full) also need to delete the key in map, which can only
// get via node - if only store value in node, cannot get key, hence cannot delete key from map
```

Time O(1) for both.

### Collection of Binary Search Operations

1. How to add an integer to every node of BST

```java
void plusOne(TreeNode root)
  if (root == null) return;
  root.val += 1;

  plusOne(root.left);
  plusOne(root.right);
```

2. How to determine if two BST identical

```java
boolean isSameTree(TreeNode root1, TreeNode root2)
  // if null, they are identical obviously
  if (root1 == null && root2 == null) return true;
  // if one is void, but the other not null, differ
  if (root1 == null || root2 == null) return false;
  // if both not void, but values diff, diff
  if (root1.val != root2.val) return false;

  // To recursively compare all pairs of node
  return isSameTree(root1.left, root2.left)
      && isSameTree(root1.right, root2.right);
```

- Simple to see the two examples as framework for handling BST.
- BST additionally satisfies the binary search property, which states that the key in each node must be greater than or equal to any key stored in the left sub-tree, and less than or equal to any key stored in the right sub-tree.

Compliance checking BST

```java
// false BST checking
boolean isValidBST(TreeNode root) {
    if (root == null) return true;
    if (root.left != null && root.val <= root.left.val) return false;
    if (root.right != null && root.val >= root.right.val) return false;

    return isValidBST(root.left)
        && isValidBST(root.right);
}

// because the requirement is for all nodes under each

// Idea: use auxiliary function to add parameters in the parameter list
// which can carry out more useful info.
boolean isValidBST(TreeNode root)
  return isValidBST(root, null, null)

boolean isValidBST(TreeNode root, TreeNode min, TreeNode max)
  if (root == null) return true;
  if (min != null && root.val <= min.val) return false;
  if (max != null && root.val >= max.val) return false;
  return isValidBST(root.left, min, root) 
      && isValidBST(root.right, root, max);

// Lookup
boolean isInBST(TreeNode root, int target)
  if (root == null) return false;
  if (root.val == target) return true;

  if (root.val < target)
    return isInBST(root.right, target)
  if (root.val > target)
    return isInBST(root.left, target)

// Delete

// Need to keep the property of BST
// Case 1: Node A is leaf node, delete directly
if (root.left == null && root.right == null)
  return null
// Case 2: Node A has only one child, then change its child nodes to its place
if (root.left == null) return root.right;
if (root.right == null) return root.left;
// Case 3: Node A has two children, A must find max node in left sub-tree or the 
// min in right sub-tree to replace its place
if (root.left != null && root.right != null)
  // Find min node in right sub-tree
  TreeNode minNode = getMin(root.right);
  // replace root node to minNode
  root.val = minNode.val;
  // Delete root node subsequently
  root.right = deleteNode(root.right, minNode.val);

// Full
TreeNode deleteNode(TreeNode root, int key) {
    if (root == null) return null;
    if (root.val == key) {
        // These two IF function handle the situation 1 and situation 2
        if (root.left == null) return root.right;
        if (root.right == null) return root.left;
        // Deal with situation 3
        TreeNode minNode = getMin(root.right);
        root.val = minNode.val;
        root.right = deleteNode(root.right, minNode.val);
    } else if (root.val > key) {
        root.left = deleteNode(root.left, key);
    } else if (root.val < key) {
        root.right = deleteNode(root.right, key);
    }
    return root;
}

TreeNode getMin(TreeNode node) {
    // The left child node is the minimum
    while (node.left != null) node = node.left;
    return node;
}

// Caveat: wouldn't change nodes by 'root.val = minNode.val'
// Generally exchange the root and minNode by a series of slightly complicated
// linked-list ops since the value of Val may be complex in data, it's time-consuming
// to modify the value of the node. LL is pointer ops.

// On the foundation of the framework of binary tree, the abstract traversal of BST:
void BST(TreeNode root, int target) {
    if (root.val == target)
        // When you find the target, your manipulation should be written here
    if (root.val < target) 
        BST(root.right, target);
    if (root.val > target)
        BST(root.left, target);
}
```

### Special DS: Monotonic Stack

```
Input:  [2,1,2,4,3]
Output: [4,2,4,-1,-1]

The NEXT GREATER than 2 after first 2 is 4 (taller)
The NEXT GREATER than 1 after first 1 is 2
There is no number greater than 4 after the first 4, so -1
There is no number greater than 3 after the first 3, so -1
```

```java
vector<int> nextGreaterElement(vector<int>& nums) {
    vector<int> ans(nums.size()); // array to store answer
    stack<int> s;
    for (int i = nums.size() - 1; i >= 0; i--) { // put it into the stack back to front
        while (!s.empty() && s.top() <= nums[i]) { // determine by height
            s.pop(); // short one go away while blocked
        }
        ans[i] = s.empty() ? -1 : s.top(); // the first tall behind this element
        s.push(nums[i]); // get into the queue and wait for later height determination
    }
    return ans;
}

// This is a template for solving monotonic queue
// Loop scan elements from back to front, while using stack and enter stack back
// to front, exit stack front to back.
// While loop to rule out elem between two 'tall' elements. Their existence has no meaning
// since there is a 'taller' element in front of them and they cannot be considered
// as NEXT GREATER NUMBER of the subsequent elements

// Time seems O(N^2) but in fact O(N)
// There are N elements each pushed into stack once, pop once at most
// Total ops proportional to element scale N

// E.g. given [73, 74, 75, 71, 69, 72, 76, 73]
// Weather temps - return an array to compute: for each day, how many days to wait
// for a warmer temp; 0 if cannot wait for that day
// Output [1,1,4,2,1,1,0,0]
vector<int> dailyTemperatures(vector<int>& T) {
    vector<int> ans(T.size());
    stack<int> s; // here for element index，not element
    for (int i = T.size() - 1; i >= 0; i--) {
        while (!s.empty() && T[s.top()] <= T[i]) {
            s.pop();
        }
        ans[i] = s.empty() ? 0 : (s.top() - i); // get index spacing
        s.push(i); // add index，not element
    }
    return ans;
}

// To mimick ring array
int[] arr = {1,2,3,4,5};
int n = arr.length, index = 0;
while (true) {
    print(arr[index % n]);
    index++;
}
// Next is not only now the right side of current element, but also the left
// side too. Double the original array or to connect another original array at the back
vector<int> nextGreaterElements(vector<int>& nums) {
    int n = nums.size();
    vector<int> res(n); // store result
    stack<int> s;
    // pretend that this array is doubled in length
    for (int i = 2 * n - 1; i >= 0; i--) {
        while (!s.empty() && s.top() <= nums[i % n])
            s.pop();
        res[i % n] = s.empty() ? -1 : s.top();
        s.push(nums[i % n]);
    }
    return res;
}
```

Monotonic Queue - LeetCode 239

Given an array of nums, there is a sliding window of size K which is moving from the leftmost to rightmost. You can only see the k numbers in the window. Each time the sliding window moves right by one position. Return the max sliding window.

```
Input: [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]

Window position             Max
---------------             ---
[1  3  -1] -3  5  3  6  7     3
 1 [3  -1  -3] 5  3  6  7     3
 1  3 [-1  -3  5] 3  6  7     5

Note: may asssume k is always valid, 1 <= k <= input size for non-empty array
```

1. Framework

Key is how to compute max value in each 'window' at O(1) time, so that the entire time O(N)
As each sliding window moves, one number added and one removed; so if to get max value in O(1), need special "monotonic queue".

```java
// Ordinary queue
class Queue {
    void push(int n);
    // or enqueue, adding element n to the end of the line
    void pop();
    // or dequeue, remove the leader element
}

// monotonic queue
class MonotonicQueue {
    // add element n to the end of the line
    void push(int n);
    // returns the maximum value in the current queue
    int max();
    // if the head element is n, delete it
    void pop(int n);
}

vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    MonotonicQueue window;
    vector<int> res;
    for (int i = 0; i < nums.size(); i++) {
        if (i < k - 1) { // fill the first k-1 of the window first
            window.push(nums[i]);
        } else { // the window begins to slide forward
            window.push(nums[i]);
            res.push_back(window.max());
            window.pop(nums[i - k + 1]);
            // nums[i - k + 1] is the last element of the window
        }
    }
    return res;
}

// Index: 0  1  2  3  4
// nums:  1  3 -1 -3  5
//          [  k=3  ]
// 3:  pop(nums[i-k+1])  
// -3: push(nums[i])
```

2. Implement monotonic queue

```java
// Double-ended queue
class deque {
    // insert element n at the head of the team
    void push_front(int n);
    // insert element n at the end of the line
    void push_back(int n);
    // remove elements at the head of the team
    void pop_front();
    // remove element at the end of the line
    void pop_back();
    // returns the team head element
    int front();
    // returns the tail element
    int back();
}

// It's easy to impl this using linked-list
// Similar to monotonic stack, the push still adds elements to the
// end of queue, but deletes the previous elements smaller than the new element
class MonotonicQueue {
private:
    deque<int> data;
public:
    void push(int n) {
        while (!data.empty() && data.back() < n) 
            data.pop_back();
        data.push_back(n);
    }
};

// Adding the size of number repr the weight of the person, squahsing the underweight in front
// and stops until in encounters a larger magnitude

// If every elem added like so, the size will eventually decreaes in a monotonic order
int max() {
    return data.front();
}

void pop(int n) {
    if (!data.empty() && data.front() == n)
        data.pop_front();
}

// data.front() == n since queue head n need to delete may have been 'squashed', so
// don't need to delete it at this time

// Full
class MonotonicQueue {
private:
    deque<int> data;
public:
    void push(int n) {
        while (!data.empty() && data.back() < n) 
            data.pop_back();
        data.push_back(n);
    }

    int max() { return data.front(); }

    void pop(int n) {
        if (!data.empty() && data.front() == n)
            data.pop_front();
    }
};

vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    MonotonicQueue window;
    vector<int> res;
    for (int i = 0; i < nums.size(); i++) {
        if (i < k - 1) { // fill the first k-1 of the window first
            window.push(nums[i]);
        } else { // window slide forward
            window.push(nums[i]);
            res.push_back(window.max());
            window.pop(nums[i - k + 1]);
        }
    }
    return res;
}

// While push ops contains a while loop, time is not O(1), so why linear?
// The push alone is not O(1), but the overall time still O(N) since each element
// in nums is pushed_back and pop_back at most once, without any redundant ops.
```

Monotonic queue keeps the monotonicity of the queue by deleting elements when adding elements, which is equivalent to extracting the monotonically increasing (or decreasing) part of a function; while priority queue (binary heap) is equivalent to automatic sorting.

### Design Twitter

LeetCode 355 - it combines both ordered linked list and OO design.

```java
// Given API
class Twitter {

    /** user post a tweet */
    public void postTweet(int userId, int tweetId) {}

    /** return the list of IDs of recent tweets, 
    from the users that the current user follows (including him/herself),
    maximum 10 tweets with updated time sorted in descending order */
    public List<Integer> getNewsFeed(int userId) {}

    /** follower will follow the followee,
    create the ID if it doesn't exist */
    public void follow(int followerId, int followeeId) {}

    /** follower unfollows the followee,
    do nothing if the ID does not exist */
    public void unfollow(int followerId, int followeeId) {}
}

// User story API
Twitter twitter = new Twitter();

twitter.postTweet(1, 5);
// user 1 posts a tweet with ID 5

twitter.getNewsFeed(1);
// return [5]
// Remarks: because each user follows him/herself

twitter.follow(1, 2);
// user 1 starts to follow user 2

twitter.postTweet(2, 6);
// user 2 posted a tweet with ID 6

twitter.getNewsFeed(1);
// return [6, 5]
// Remarks: user 1 follows both user 1 and user 2,
// return the recent tweets from both users,
// with tweet 6 in front of tweet 5 as tweet 6 is more recent

twitter.unfollow(1, 2);
// user 1 unfollows user 2

twitter.getNewsFeed(1);
// return [5]
```

- FB, once added a friend, her recent posts show up in refreshed feeds, sorted in descending order
- The diff is Twitter is unidirectional, while FB friends are bi-directional
- Most API are easy to impl. The hard ones are `get_NewsFeed` per need to sort by time in descending
- However the list of followees are dynamic, which makes these hard to keep updated
- Algo: imagine store each user's own tweets in a linked-list sorted by timestamp, with each node as tweet's ID and timestamp; if a user follows k, we can combine these k ordered linked-list and apply an algo to get the correct `getNewsFeed`
- Also, how to use code to repr users and tweets? OO helps.

```java
class Twitter {
    private static int timestamp = 0;
    // Store tweets posted by a user
    private static class Tweet {}
    private static class User {}

    /* the APIs skeleton */
    public void postTweet(int userId, int tweetId) {}
    public List<Integer> getNewsFeed(int userId) {}
    public void follow(int followerId, int followeeId) {}
    public void unfollow(int followerId, int followeeId) {}
}

class Tweet {
    private int id;
    private int time;
    private Tweet next;

    // initialize with tweet ID and post timestamp
    public Tweet(int id, int time) {
        this.id = id;
        this.time = time;
        this.next = null;
    }
}

// User class needs to store userId, list of followers, list of posted tweets
// List of followees can use hash-set to avoid dup and fast search
// list of posted tweets as LL to merge with order
// static int timestamp = 0
class User {
    private int id;
    public Set<Integer> followed;
    // The head of the linked list of posted tweets by the user
    public Tweet head;

    public User(int userId) {
        followed = new HashSet<>();
        this.id = userId;
        this.head = null;
        // follow the user him/herself
        follow(id);
    }

    public void follow(int userId) {
        followed.add(userId);
    }

    public void unfollow(int userId) {
        // a user is not allowed to unfollow him/herself
        if (userId != this.id)
            followed.remove(userId);
    }

    public void post(int tweetId) {
        Tweet twt = new Tweet(tweetId, timestamp);
        timestamp++;
        // insert the new tweet to the head of the linked list
        // the closer a tweet is to the head, the larger the value of time
        twt.next = head;
        head = twt;
    }
}

// Impl APIs
class Twitter {
    private static int timestamp = 0;
    private static class Tweet {...}
    private static class User {...}

    // we need a mapping to associate userId and User
    private HashMap<Integer, User> userMap = new HashMap<>();

    /** user posts a tweet */
    public void postTweet(int userId, int tweetId) {
        // instantiate an instance if userId does not exist
        if (!userMap.containsKey(userId))
            userMap.put(userId, new User(userId));
        User u = userMap.get(userId);
        u.post(tweetId);
    }

    /** follower follows the followee */
    public void follow(int followerId, int followeeId) {
        // instantiate if the follower does not exist
        if(!userMap.containsKey(followerId)){
            User u = new User(followerId);
            userMap.put(followerId, u);
        }
        // instantiate if the followee does not exist
        if(!userMap.containsKey(followeeId)){
            User u = new User(followeeId);
            userMap.put(followeeId, u);
        }
        userMap.get(followerId).follow(followeeId);
    }

    /** follower unfollows the followee, do nothing if follower does not exists */
    public void unfollow(int followerId, int followeeId) {
        if (userMap.containsKey(followerId)) {
            User flwer = userMap.get(followerId);
            flwer.unfollow(followeeId);
        }
    }

    /** return the list of IDs of recent tweets, 
    from the users that the current user follows (including him/herself),
    maximum 10 tweets with updated time sorted in descending order */
    public List<Integer> getNewsFeed(int userId) {
        // see below as we need to understand the algorithm
    }
}
```

Design of the Algo

- combines k-ordered linked list is implemented using Priority Queue
- sorted by time in descending (larger timestamp means more recent)

```java
public List<Integer> getNewsFeed(int userId)
  List<Integer> res = new ArrayList<>();
  if (!userMap.containsKey(userId)) return res;
  // IDs of followees
  Set<Integer> users = userMap.get(userId).followed;
  // auto sorted by time in descending
  PriorityQueue<Tweet> pq = new PriorityQueue<>(users.size(), (a, b)->(b.time - a.time));

  // first, insert all heads of linked list into the PQ
  for (int id : users)
    Tweet twt = userMap.get(id).head;
    if (twt == null) continue;
    pq.add(twt);

  while (!pq.isEmpty())
    // return only 10 records
    if (res.size() == 10) break;
    // pop the tweet with the larggest item (most recent)
    Tweet twt = pq.pop();
    res.add(twt.id);
    // insert the next tweet, which will be sorted
    if (twt.next != null)
      pq.add(twt.next);

  return res;
```

- A simple timeline function using OO pattern and algo combining k-sorted linked lists
- User and Tweet classes with APIs
- Not for scale - a lot more details in read and write performance of DB, limit of memory cache
- Real apps are big and complicated engineering projects

```
Client - DNS - CND
Load Balancer
Web Server
Read API - Tweet Info Service - Timeline Service - User Info Service
Write API - Search API, Fan Out - Fan Out Service - User Graph Service - Notification Service
Memory Cache
SQL Read Replica - SQL Write Master-Slave - Object Store
```

### Reverse Part of Linked List via Recursion

```java
// node structure for a SLL
public class ListNode
  int val;
  ListNode next;
  ListNode(int x) { val = x; }
```

- Double loop is a base solution - first-loop find m-th, then another to reverse entries between m and n
- Need to be careful with detail, keeping the SLL integrity
- Recursion is more elegant

```java
ListNode reverse(ListNode head)
  if (head.next == null) return head;
  ListNode last = reverse(head.next);
  head.next.next = head;
  head.next = null;
  return last

// The key for recursion is to clarify the definition of the recursive function.
// Here: input a node `head`, we will reverse the list starting from `head`, and return the new head node

// 1 (head) -> reverse(2 -> 3 -> 4 -> 5 -> 6 -> NULL)
// after reverse(head.next), the whole LL becomes
// 1 (head) -> NULL <- 2 <- 3 <- 4 <- 5 <- 6 (last)
// Per definition, reverse() -> new head node, so use `last` to mark it
// so head.next.next = head;
// 1 (head) <- 2 ... (change NULL pointer to current head 1)
// head.next = null;
// NULL <- 1 (head) <- 2 ...

// Some key points: 
// 1) Recursion needs base case
// if (head.next == null) return head;
// which means if only one node, after reversion, the head is still itself
// 2) after reversion, new head is `last` and former `head` becomes last node, so ensure to point its tail to null
// head.next = null;
```

Reverse first N nodes

```
reverse(head, 3); N = 3

result:

1 <- 2 <- 3
|
 -> 4 -> 5 -> 6 -> NULL

(3 returned as new head or last)
```

```java
ListNode successor = null;

// reverse n nodes starting from head, return new head
ListNode reverseN(ListNode head, int n)
  if (n == 1)
    // mark (n + 1)-th node
    successor = head.next;
    return head;
  // starts from head.next
  ListNode last = reverseN(head.next, n - 1);

  head.next.next = head;
  // link new head to successor
  head.next = successor;
  return last;

// Base case n == 1, if only one element, then new head is itself, meanwhile ensure marking successor node
// previously head.next = null because after reversing the whole list, head becomes the last node;
// but now head may not be, so need to mark successor n+1 and link it to `head`
// 1 (head) <- 2 <- 3 (last)
// |
// -> 4 (successor) -> ...
```

Reverse m to n node

```java
// If m == 1, it's equal to reversing the first n
ListNode reverseBetween(ListNode head, int m, int n)
  // base
  if (m == 1)
    return reverseN(head, n);

  // Else: if taking index of head as 1, then need to reverse from m-th
  // If taking head.next as index 1, then compared to head.next, the reverse section should
  // start from (m - 1), and what about head.next.next ...
  // Diff from iteration, this is how we think in recursive way
  head.next = reverseBetween(head.next, m - 1, n - 1);
  return head;
```

### Queue-Stack and Stack-Queue

Stacked-Queue

```java
// Two opposing queue, facing outwards from each other

// Front            Rear
// --------   ----------
//         | |
// --------   ----------
//    s2         s1

class MyQueue {
    private Stack<Intager> s1, s2;

    public MyQueue() {
        s1 = new Stack<>();
        s2 = new Stack<>();
    }

    /** Push element x to the back of queue. */
    public void push(int x) {
        s1.push(x;)
    }

    /** Removes the element from in front of queue and returns that element. */
    public int pop() {
      // first call peek() to ensure s2 not empty
      peek();
      return s2.pop();
    }

    /** Get the front element. */
    public int peek() {
        if (s2.isEmpty())
          // move all s1 to s2
          while (!s1.isEmpty())
            s2.push(s1.pop())
        return s2.peek()
    }

    /** Returns whether the queue is empty. */
    public boolean empty() {
      return s1.isEmpty() && s2.isEmpty();
    }
}
```

- Time O(N) worst case - push(N) then pop()
- Time O(1) on average - for an entry it can only be moved at most once, which means that the average time complexity of each element of `peek()` is O(1)
