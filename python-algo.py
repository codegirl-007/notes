# Time Complexity: O(n)
# Best used when data is not sorted
def linear_search(arr, point):
    for i in arr:
        if i == point:
            return True
    return False

# Time Complexity: O(log n)
# Best used when you have sorted data, works great on large data sets
def binary_search(arr, point):
    first = 0
    last = len(arr) - 1
    
    while last >= first:
        mid = (first + last) // 2
        if arr[mid] == point:
            return True
        else:
            if point < arr[mid]:
                last = mid - 1
            else:
                first = mid + 1
    return False

# Time Complexity: O(n^2)
# Best for smaller data sets
# Stable sort
def bubble_sort(arr):
    list_length = len(arr) - 1
    for i in range(list_length):
        for j in range(list_length):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j] # swap
    return list
    
# Time Complexity: O(n^2)
# Best used in nearly sorted data. When list is nearly sorted or sorted, time complexity can be O(n)
def insertion_sort(arr):
    for i in range(1, len(arr)): # start with the second item because a the first item will always be sorted
        value = arr[i]
        while i > 0 and arr[i - 1] > value: # while we are not at the start of the list and prev is greater than current
            arr[i] = arr[i - 1] # swap part 1
            i = i - 1 # decrement
        arr[i] = value # swap part 2
    return arr

# Time complexity: O(n *log n) because the list does split into smaller list but requires linear seartime to handle each item in sublists to merge them
# One of the most efficient sorting algorithms
# Stable
def merge_sort(arr):
    if len(arr) > 1: # if list length is greater than 1, base case (we're breaking the list down to 1 element each)
        mid = len(arr) // 2 # find the mid point
        left_half = arr[:mid] # and split your lists in half
        right_half = arr[mid:]
        merge_sort(left_half) # sort your left half
        merge_sort(right_half) # sort your right half
        
        # this section is merging the two lists
        left_ind = 0 # track the index of the left half
        right_ind = 0 # track index of right half
        alist_ind = 0 # track index of arr
        while left_ind < len(left_half) and right_ind < len(right_half):
            if left_half[left_ind] <= right_half[right_ind]: # if left is less than right
                arr[alist_ind] = left_half[left_ind] # put left in arr
                left_ind += 1 # increment left index
            else:
                arr[alist_ind] = right_half[right_ind] # if left is greater than right, put the right in list
                right_ind += 1 # increment right
            alist_ind += 1 # increment the index which becomes the next index we place into
            
        # take care of the remainders
        while left_ind < len(left_half): # if the index has not reached the end of list 
            arr[alist_ind] = left_half[left_ind] # place items in list
            left_ind += 1 # increment
            alist_ind += 1 # increment
            
        while right_ind < len(right_half): # if index has not reached end of list
            arr[alist_ind] = right_half[right_ind]  # add to list
            right_ind += 1 # increment
            alist_ind += 1 # increment

# Time Complexity: O(nlogn) on average, worst case O(n^2)
# Unstable
def quick_sort(arr, start, end):
    if (end - start + 1 <= 1):
      return arr
    
    pivot = arr[end]
    left = start

    # Move elements smaller than the povot to the left side of the pivot
    for i in range(start, end):
        if (arr[i] < pivot):
            tmp = arr[left]
            arr[left] = arr[i]
            arr[i] = tmp
            left += 1

    # Move pivot in-between left & right sides
    arr[end] = arr[left]
    arr[left] = pivot

    # Quicksort left side
    quick_sort(arr, start, left - 1)

    # Quicksort right side
    quick_sort(arr, left + 1, end)

    return arr

# Time Complexity: O(n) even in the worst case
# Rarely ever used
# Ideal when you know the items you're sorting fit in a finite (and small) range
# Unstable
def bucket_sort(arr):
    counts = [0, 0, 0] # here we assume that our list only has 0, 1, or 2

    for n in arr:
        counts[n] += 1
      
    i = 0
    for n in range(len(counts)):
        for j in range(counts[n]):
            arr[i] = n
            i += 1
    return arr

# Two strings are anagrams if they contain the same letters but not in the same order (case does not matter)
# Strategy is to sort them and see if the sorted letters between strings are the same
def is_anagram(s1, s2):
    s1 = s1.replace(' ', '').lower() # replace spaces and normalize to lower
    s2 = s2.replace(' ', '').lower() # replace spaces and normalize to lower
    
    if sorted(s1) == sorted(s2): # sorted will let you sort any iterable
        return True
    else:
        return False

def is_palindrome(s1):
    if s1.lower() == s1[::-1].lower(): # lowercase to normalize, compare the string to the reversed string
        return True
    return False

def last_digit(str):
    return [c for c in str if c.isdigit()[-1]] # filter everything that isn't a digit

# Time complexity O(n)
# import string
def cipher(str, key): # accept a string you want to encrypt and key being the number of places you are shifting each letter
    uppercase = str.ascii_uppercase # a string of all the lowercase letters
    lowercase = str.ascii_lowercase # a string of all the uppercase letters
    encrypt = '' # your soon to be encrypted string

    for c in str:
        if c in uppercase: # if the character is in uppercase 
            new = (uppercase.index(c) + key) % 26 # take the index and add the key; we use %26 to handle the edge case of Z, we want to start at 0 and add key to not go out of bounds
            encrypt += uppercase[new] # append to the encrypted string
        elif c in lowercase: # do the same for lowercase
            new = (lowercase.index(c) + key) % 26
            encrypt += lowercase[new]
        else:
            encrypt += c

    return encrypt

# Fizz Buzz challenge: 
# Write a program that prints the numbers from 1 to 100. If the number is a multiple of 3, print “Fizz.” 
# If the number is a multiple of 5, print “Buzz.” If the number is a multiple of 3 and 5, print “FizzBuzz.”
def fizz_buzz(n):
    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 == 0:
            print('FizzBuzz')
        elif i % 3 == 0:
            print('Fizz')
        elif i % 5 == 0:
            print('Buzz')
        else:
            print(i)

# Greatest common factor
# Time complexity: O(n) as number of steps gets bigger if i gets bigger
# This iteration has a boundary condition that is unable to handle 0
def gcf(i1, i2):
    gcf = None
    if i1 < 0 or i2 < 0:
        raise ValueError("Numbers must be positive.")
    if i1 == 0:
        return i2
    if i1 == 0:
        return i
    if i1 > i2:
        smaller = i2
    else:
        smaller = i1

    for i in range(1, smaller + 1):
        if (i1 % i == 0 and i2 % i == 0):
              gcf = i
    return gcf

# Euclid's Algorithm is more efficient than the gcf above.
# Time complexity: O(log(n))
def gcf(x, y):
    if y == 0: 
        x, y = y, x # address the boundary condition. If y is = 0, you swap x and y to address it.

    while y != 0:
        x, y = y, x % y # here we are finding the remainder and making the remainder x
    
    return x

# Time complexity: O(n)
def is_prime(n):
    for i in range(2, n):
        if n%i == 0:
            return False
        
    return True

# is_prime can be improved
# import math
# Time complexity: O(sqrt(n))
def is_prime(n):
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
          return False
    return True

# Move zeros to the end
def move_zeros(arr):
    zero_index = 0
    for index, n in enumerate(arr): # loop through every number in arr, enumerate to track both the index and the current number
        if n != 0:
            arr[zero_index] = n # if n is not zero, use the index stored to replace whatever is at zero index
            if zero_index != index:
              arr[index] = 0 # zero does not equal the index which means there was a zero earlier in the list so put a zero at the current spot
            zero_index += 1 # increment zero  
    return(arr)

# find duplicates in a list
def return_dups(arr):
    dups = [] # create an array to hold dupes
    a_set = set() # create a set for uniques

    for item in arr:
        l1 = len(a_set) # take the initial length of a set at the start of an iteration
        a_set.add(item) # add an item to the set
        l2 = len(a_set) # take the length again
        if l1 == l2:
            dups.append(item) # if the length at the start and after the addition are the same length, it means the current item is a dup
    return dups # return the array


# find the intersection of two list
def find_intersection(arr1, arr2):
    list3 = [v for v in arr1 if v in arr2] # add value to list if value is in the second list
    return list3

def find_intersection_1(arr1, arr2):
    set1 = set(arr1) # convert list to set
    set2 = set(arr2) 
    return list(set1.intersection(set2)) # intersection will find the duplicates and then change it back into a list

# Create a linked List
# The disadvantage of a linked list is that you can only get a node is by iterating whereas in an array, you can access it in constant time
# However, the advantage of Linked List is that adding a node is in O(1) time where as it would be O(n) for an array
# Linked Lists needs more memory than an array because you have to keep a pointer. If the data is small, your list can be twice the size of an array
# Linked Lists also don't allow random access meaning you can't access it in constant time.
class Node: 
    def __init__(self, data, next=None):
        self.data = data
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        if not self.head: # if your list does not have a head, make a node and set it to your head
            self.head = Node(data) 
            return # return out of it
        current = self.head # otherwise, set current to head
        while current.next: # loop through the linked list
            current = current.next # until you get to a node that does not have a next
        current.next = Node(data) # add node to end

    def __str__ (self): # give the ability to print the list
        node = self.head # take the first node
        while node is not None: # iterate
            print(node.data) # print data
            node = node.next # move to next node

    # find a node that has what you're looking for. Done by iterating and seeing if the data matches the target
    def search(self, target):
        current = self.head
        while current.next:
            if current.data == target:
                return True
            else:
                current = current.next
        return False
    
    def remove(self, target):
        if self.head == target: # first handle what happens if the node you want to delete is your head
            self.head = self.head.next # set your head to the next item, the original head will be gc'd
            return
        current = self.head # otherwise you iterate through
        previous = None

        while current:
            if current.data == target: # once you find your target
                previous.next = current # set the previous.next (current) to current.next (the next node), effectively dropping the current node
                current = current.next # then move your current to the next one

    def reverse(self):
        current = self.head # place yourself at the beginning of the list
        previous = None # keep track of your previous 
        while current: # iterate through the list
            next = current.next # save the current's next
            current.next = previous # set the current.next to the previous
            previous = current # save the current node to previous
            current = next # set current to the next one
        self.head = previous # set your head to previous... basically move your pointers around and storing data so that you don't lose anything.

    def is_ring(self):
        slow = self.head
        fast = self.head

        # use an infinite loop and loop through both slow and fast, fast being one step ahead of slow. If the two ever equal the same node, it is a ring
        while True: 
            try:
                slow = slow.next
                fast = fast.next.next
                if slow is fast:
                  return True
            except:
                return False
            
# Stacks
# Bounded stacks: has a limit on how many items you can add to it
# Unbounded stacks: has no limit
class Stack:
    def __init__(self):
        self.items = []

    def push(self, data):
        self.items.append(data)

    def pop(self):
        return self.items.pop()
    
    def size(self):
        return len(self.items)
    
    def is_empty(self):
        return len(self.items) == 0
    
    def peek(self):
        return self.items[-1] # take the last item from the list
    
# You can also use a Linked List to represent a stack

class Stack:
    def __init__(self):
        self.head = None

    def push(self, data):
        node = Node(data)
        if self.head is None: # if your list doesn't have a head, you create a node and set it to thead
            self.head = node
        else: # otherwise you make your new node the head
            node.next = self.head
            self.head = node

    def pop(self):
        if self.head is None: # if there is no head then the list is empty
            raise IndexError('pop from empty stack')
        poppednode = self.head # otherwise you can return self.head but you need to store it
        self.head = self.head.next # so that you can set it to the next node
        return poppednode.data
    
# use stacks to reverse a string
def reverse_str(str):
    stack = [] # make your stack
    string = "" # make your answer string
    for char in str: # iterate
        stack.append(char) # push each character to stack
    for char in str: # for the length of your original string
        string += stack.pop() # pop from your stack into your answer string
    return string
    
# Min stack - have the ability to return the smallest element
class MinStack():
    def __init__(self):
        self.main = [] # main is to keep track of your main stack
        self.min = [] # min is to keep track of the smallest elemetn

    def push(self, n):
        if len(self.main) == 0: # check to see if main is empety because if it is, no matter what n is, it is the smallest number
            self.min.append(n)
        elif n <= self.min[-1]: # if not empty, check whether n is less than or equal to the last item in min
            self.min.append(n) # if it is, then append n
        else: # otherwise append the last item to min again which keeps the number of items in sync with the main stack
              self.min.append(self.min[-1])
        self.main.append(n)

    def pop(self):
        self.min.pop()
        return self.main.pop()
    
    def get_min(self):
        return self.min[-1]
    
  # valid parentheses
def check_parens(str):
  stack = []
  for c in str:
      if c == "(":
          stack.append(c)
      if c == ")":
          if len(stack) == 0:
              return False
          else:
              stack.pop()
  return len(stack) == 0

# Queues
# Queues are not efficient for accessing individual pieces of data because it is O(n) time (you have to iterate)
# Ideal for anything that is first come, first served

class Node: # basically the same as the linked list node
    def __init__(self, data, next=None):
        self.data = data
        self.next = next
    
class Queue:
    def __init__(self):
        self.front = None # track the front and rear so that you can enqueue and dequeue in constant time O(n)
        self.rear = None
        self._size = 0 # used for bookkeeping the size of queue

    def enqueue(self, item): # this will add items to to rear
        self._size += 1 # increase the size
        node = Node(item) # create a node of the item
        if self.rear is None: # if there is no rear
            self.front = node # then the list is empty so you set the front and rear to your new node
            self.rear = node
        else: # but if there is a rear
            self.rear.next = node # append your new node to the current self.rear
            self.rear = node # then point self.rear to your new node

    def dequeue(self):
        if self.front is None: #  if there is nothing in your queue, then throw an error
            raise IndexError("pop from empty queue")
        self._size -= 1 # otherwise decrease the size
        temp = self.front # store the item you are going to remove
        self.front = self.front.next # pont your front to the next node
        if self.front is None: # if none happens to be nothing
            self.rear = None # then set your rear to nothing because your list is empty
        return temp.data # then return the item you removed
    
    def size(self):
        return self._size
    
# Queue using two stacks
class Queue:
    def __init__(self):
        self.s1 = []
        self.s2 = []

    def enqueue(self, item):
        while len(self.s1) != 0: # while the length of your first stack is not empty
          self.s2.append(self.s1.pop) # pop it off into the second stack
        self.s1.append(item) # add the item to the first stack
        while len(self.s2) != 0: # then add everything back
            self.s1.append(self.s2.pop)

    def dequeue(self):
        if len(self.s1) == 0:
            raise Exception("Cannot pop from empty queue")
        return self.s1.pop()
    
    def size(self):
        return len(self.s1)

# Hash Tables
# Unlike other data structures, searching for data in a hash table is O(1) constant time
# Inserting and deleting is also O(1) time
# Efficiency is eroded when collisions happen which can make search, insert and delete O(n) time in the worse case scenario

# count all the characters in a string
def count(str):
    cdict = {}
    for char in str:
        if char in cdict: # if the char is already in the dict, increment what is there
            cdict[char] += 1
        else: # or else add it to the dict and set it to 1
            cdict[char] = 1
    print(cdict)

# two sum using hash tables (return the indexes of the number that add up to the target value)
def two_sum(nums, target):
    ndict = {}
    for index, n in enumerate(nums): # track the index and the n
        rem = target - n # subtract n from target number, giving you the remainder
        if rem in ndict: # if remainder is in the dict
            return index, ndict[rem] # return the index and then look up the index of the other number
        else:
            ndict[n] = index

# binary trees
# each node can only have two children
# every node except the parent is either the right or left child

# binary search tree is a data structure where each node can only have two children
# stores the node in sorted order such that any node has a value greater than it's left sub tree
# cannot duplicate values in a binary search tree, can get around this by adding a count value to each node and incrementing when there is a dupe
# cannot always traverse a tree without backtracking

# insert, delete and search are all O(n) in a general and binary tree
# bst are more efficient because the three operations are O(log n) time

# trees are great for storing data in a hierarchy such as a file structure or html

# create a binary tree
class BinaryTree:
    def __init__(self, value):
        self.key = value # holds the data
        self.left_child = None # left child
        self.right_child = None # right child

    def insert_left(self, value):
        if self.left_child == None: # if there is no sub tree 
            self.left_child = BinaryTree(value) # insert one with your value
        else: # otherwise
            bin_tree = BinaryTree(value) # create a tree
            bin_tree.left_child = self.left_child # assign the object currently as the left child to the binary tree you created
            self.left_child = bin_tree # then point self.left child to the bin tree

    def insert_right(self, value): # same dealio happens here except the right is taken care of
        if self.right_child == None:
            self.right_child = BinaryTree(value)
        else:
            bin_tree = BinaryTree(value)
            bin_tree.right_child = self.right_child
            self.right_child = bin_tree
    
    # alternative insert method where you are given the root and the value
    def insert(self, value):
        if not self:
            return BinaryTree(value)
        
        if value > self.key:
            self.right_child = self.insert(self.right_child, value)
        elif value < self.key:
            self.left_child = self.insert(self.left_child, value)
        return self

    # breadth-first tree traversal that returns true/false if a node contains the value (key)
    # start at the root and go level by level through your tree
    # visit each node until you reach the final level
    def breadth_first_search(self, n):
      current = [self] # keep track of nodes in the current level you are searching
      next = [] # keep track of nodes at the second level
      while current: # while there are still nodes to search at the current level
          for node in current: # iterate through every node in the current level
              if node.key == n: # if node matches n
                  return True # return true
              if node.left_child: # otherwise if node has a left child
                  next.append(node.left_child) # append it to the next array
              if node.right_child: # if it has right children, also append to next array
                  next.append(node.right_child)
          current = next # if you haven't found it yet, set your next to current
          next = [] # empty out next
      return False # if nothing is found, return false
    
    # Alternative search method where you are passing in the root node and returning the node that contains the value
    def searchBinaryTree(self, val):
        if not self or self.val == val:
            return self

        if val > self.val:
            return self.searchBinaryTree(self.right, val)
        elif val < self.val:
            return self.searchBinaryTree(self.left, val)
        else:
            return self.val
    
    # depth first tree traversal
    # visit all nodes in a binary tree
    # go as deep as you can in one direction before moving to the next sibling
    # there are 3 ways to visit every node: preorder, post order and in order

    # preorder : start with the root and move to the left and then to the right
    def preorder(self, tree):
      if tree: # recursively call yourself until you hit a base case
          print(tree.key) # print the value of every node
          self.preorder(tree.left_child) 
          self.preorder(tree.right_child) 
    
    # post order: move through a tree starting on the left, then moving right, and ending with the starting root
    def postorder(self, tree):
        if tree:
            self.postorder(tree.left_child)
            self.postorder(tree.right_child)
            print(tree)

    # inorder : print the node's value in between your two recursive calls
    # move through the tree from left to the root to the right
    def inorder(self, tree):
        if tree:
          self.inorder(tree.left_child)
          print(tree.key)
          self.inorder(tree.right_child)

    # invert a binary tree: swap all the nodes in it. left becomes right, right becomes left
    # visit every node on it  and keep track of each node's children so you can swap them
    def invert(self):
        current = [self]
        next = []
        while current: # keep track of the current level and the next level like you did in bfs
            for node in current:
                if node.left_child:
                    next.append(node.left_child)
                if node.right_child:
                    next.append(node.right_child)
                tmp = node.left_child # use a tmp var so you can perform a swap of the left and right nodes
                node.left_child = node.right_child
                node.right_child = tmp
            current = next
            next = []

    def minValueNode(self):
        curr = self.right_child
        while curr and curr.left_child:
            curr = curr.left_child
        return curr
    
    def remove(self, val):
        if not self:
            return None
        
        if val > self.key:
            self.right_child = self.remove(self.right_child, self.key)
        elif val < self.key:
            self.left_child = self.remove(self.left_child, self.key)
        else:
            if not self.left_child:
                return self.right_child
            elif not self.right_child:
                return self.left_child
            else:
                minNode = self.minValueNode(self.right_child)
                self.key = minNode.key
                self.right_child = self.remove(self.right_child, minNode.key)
        return self

# Binary heaps

# Heap : tree based data structure in which each node keeps track of two pieces of information: value and priority
# Binary heap is a heap you build using a complete binary tree (only the last level is allowed to have a missing node)
# There are two types of binary heaps: max heap and min heaps

# Max heap: parent's node's priority is always greater than or equal to any child node's priority
# Node with the highest priority is the tree root

# Min heap: parent node's priority is always less than or equal to the child node's priority
# Node with the lowest priority is at the root of the tree

# left child = 2 * index
# right child = 2 * index + 1
# parent = i / 2

class Heap:
    def __init__(self):
        self.heap = [0] # this is dummy value

    def push(self, val):
        self.heap.append(val)
        i = len(self.heap) - 1

        # the time complexity is O(log n)
        while self.heap[i] < self.heap[i // 2]: # while the newly added item is less than the parent
            tmp = self.heap[i] # store item in temp variable
            self.heap[i] = self.heap[i // 2] # do a swap
            self.heap[i // 2] = tmp
            i = i // 2

    # remove the root and then replace it with the last value in our tree in order to maintain the structure property
    # then we need to fix the order property
    def pop(self):
      if len(self.heap) == 1:
          return None
      if len(self.heap) == 2:
          return self.heap.pop()
      
      res = self.heap[1]
      self.heap[1] = self.heap.pop()
      i = 1 # set pointer to root node

      # time complexity would be O(log n) because it is percolating down the height of the tree
      while 2 * i < len(self.heap):
          if (2 * i + 1 < len(self.heap) and
              self.heap[2 * i + 1] < self.heap[2 * i] and 
              self.heap[i] > self.heap[2 * i + 1]):
              # swap the right child
              tmp = self.heap[i]
              self.heap[i] = self.heap[2 * i + 1]
              self.heap[2 * i + 1] = tmp
              i = 2 * i + 1
          elif self.heap[i] > self.heap[2 * i]:
              # swap left child
              tmp = self.heap[i]
              self.heap[i] = self.heap[2 * i]
              self.heap[2 * i] = tmp
              i = 2 * i
          else:
              break
      return res

# Graphs: abstract data type in each a piece of data connects to one or more other pieces of data
# Each piece of data in graph is called a vertext and a vertext has a key
# Vertex can have additional data called a payload
# connection between vertices in a graph is called an edge
# A graph's edce can contain a weight which is the cost to travel between vertices

# Directed Graph : graph in which each edge has a direction associated with it, you can move betwen two vertices only in that driection
#                  but you can also make a two way connection
# Great choice for creating a graph representing a social network wtih follows

# Undirected Graph: graph in which edges are bidriectional (can travel in either direction)

# Complete graph: graph in which every vertext is connected to every other vertex

# Incomplete graph: some but not all vertices are connected

# Path is a sequence of vertices connected by edges
# Cycle is a path in a graph that starts and ends at the same vertex
# Acyclic graph is one that does not contain a cycle
# Edge list is the data structure where you represent each edge in a graph with two vertices that connect

# Example: 
"""
[
[10, 20]
[10, 30]
[20, 10]
[20, 30]
[30, 10]
[30, 20]
[30, 40]
[40, 30]
]
"""

# adjacency list is a connection of unordered lists, wiht each list representing the connections for a single vertex

# Example:
"""
{
10: [20, 30],
20: [10, 30],
30: [10, 20, 40],
40: [30]
}
"""

# define a vertex class
class Vertex: # also called a node
    def __init__(self, key):
        self.key = key
        self.connections = {} # dictionary where you will store the vertices each vertex is adjacent to

    def add_adj(self, vertex, weight = 0): # makes a vertex adjacent to the vertex this is called on
        self.connections[vertex] = weight # by adding a connection

    def get_connections(self):
        return self.connections.keys()
    
    def get_weight(self, vertex):
        return self.connections[vertex]
    
class Graph:
    def __init__(self):
        self.vertex_dict = {} # stores the vertices

    def add_vertex(self, key): # adds a new vertex
        new_vertex = Vertex(key) # create new vertex
        self.vertex_dict[key] = new_vertex # map key to the new vertex

    def get_vertex(self, key): # use the key to see if the vertex is in graph
        if key in self.vertex_dict[key]:
            return self.vertex_dict[key]
        return None
    
    def add_edge(self, f, t, weight = 0): # adds an edge between two vertices in graph
        if f not in self.vertex_dict:
            self.add_vertex(f)
        if t not in self.vertex_dict:
            self.add_vertex(t)
        self.vertex_dict[f].add_adj(self.vertex_dict, weight)
        
# Dijkstra's Algorithm
# Used to find the shortest path from a vertex in a graph to every other vertex

# pick a starting vertex

#import heapq
 
 
def dijkstra(graph, starting_vertex):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[starting_vertex] = 0
    pq = [(0, starting_vertex)]
 
    while len(pq) > 0:
        current_distance, current_vertex = heapq.heappop(pq)
        if current_distance > distances[current_vertex]:
            continue
 
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances