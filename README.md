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

## Common Knowledge

### Linux Process and Thread

### Linux Shell

### Cookies and Session

### Cryptography

### Online Practice

Git - https://learngitbranching.js.org

SQL - sqlzoo.net

