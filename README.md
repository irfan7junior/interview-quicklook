# Interview Quicklook

---

- [Interview Quicklook](#interview-quicklook)
  - [Big-O Notation](#big-o-notation)
    - [Different Charts](#different-charts)
  - [Array](#array)
    - [Dynamic Array](#dynamic-array)
    - [Amortized Analysis](#amortized-analysis)
    - [Problems](#problems)
      - [Problem](#problem)
      - [Solution](#solution)
      - [Problem](#problem-1)
      - [Solution](#solution-1)
      - [Problem](#problem-2)
      - [Solution](#solution-2)
      - [Problem](#problem-3)
      - [Solution](#solution-3)
      - [Problem](#problem-4)
      - [Solution](#solution-4)
      - [Problem](#problem-5)
      - [Solution](#solution-5)
      - [Problem](#problem-6)
      - [Solution](#solution-6)
  - [Stack, Queue, Deque](#stack-queue-deque)
    - [What is a Stack](#what-is-a-stack)
    - [What is a Queue](#what-is-a-queue)
    - [What is a Deque](#what-is-a-deque)
    - [Problems](#problems-1)
      - [Problem](#problem-7)
      - [Solution](#solution-7)
      - [Problem](#problem-8)
      - [Solution](#solution-8)
  - [Linked List](#linked-list)
    - [What is Linked List](#what-is-linked-list)
      - [Pros](#pros)
      - [Cons](#cons)
    - [What is Doubly Linked List](#what-is-doubly-linked-list)
    - [Problems](#problems-2)
      - [Problem](#problem-9)
      - [Solution](#solution-9)
      - [Problem](#problem-10)
      - [Solution](#solution-10)
      - [Problem](#problem-11)
      - [Solution](#solution-11)
  - [Recursion](#recursion)
    - [Problems](#problems-3)
      - [Problem](#problem-12)
      - [Solution](#solution-12)
      - [Problem](#problem-13)
      - [Solution](#solution-13)
      - [Problem](#problem-14)
      - [Solution](#solution-14)
    - [Memoization](#memoization)
    - [Problems](#problems-4)
      - [Problem](#problem-15)
      - [Solution](#solution-15)
      - [Problem](#problem-16)
      - [Solution](#solution-16)
      - [Problem](#problem-17)
      - [Solution](#solution-17)
      - [Problem](#problem-18)
      - [Solution](#solution-18)
  - [Dynamic Programming](#dynamic-programming)
    - [Memoization Recipe](#memoization-recipe)
    - [Fibonacci Problem](#fibonacci-problem)
    - [Grid Traveler Problem](#grid-traveler-problem)
    - [Can Sum DP Type](#can-sum-dp-type)
    - [How Sum DP Type](#how-sum-dp-type)
    - [Best Sum DP Type](#best-sum-dp-type)
    - [In a Nutshell](#in-a-nutshell)
    - [Can Construct DP Type](#can-construct-dp-type)

## Big-O Notation

### Different Charts

<details>

  <summary>Complexity Chart</summary>

![](mdImages/2021-02-17-12-33-06.png)

</details>

<details>

  <summary>Common Data Structure Operations</summary>

![](mdImages/2021-02-17-12-46-17.png)

</details>

<details>

  <summary>Various Sorting Algorithms</summary>

![](mdImages/2021-02-17-12-46-57.png)

</details>

<details>

  <summary>Python List Big-O Complexity</summary>

![](mdImages/2021-02-17-12-49-15.png)

</details>

<details>

  <summary>Python Dictionary Big-O Complexity</summary>

![](mdImages/2021-02-17-12-50-04.png)

</details>

<details>

  <summary>Ploting</summary>

```py
import math
from matplotlib import pyplot
import numpy as np

pyplot.style.use('bmh')

# Setup runtime comparisions
n = np.linspace(1, 10, 1000)
labels = ['Constant', 'Logarithmic', 'Linear', 'Log Linear', 'Quadratic', 'Cubic', 'Exponential']
big_o = [np.ones(n.shape), np.log(n), n, n*np.log(n), n**2, n**3, 2**n]

# Plot setup
pyplot.figure(figsize=(12, 10))
pyplot.ylim(0, 50)
for i in range(len(big_o)):
  pyplot.plot(n, big_o[i], label=labels[i])

pyplot.legend(loc=0)
pyplot.ylabel('Relative Runtime')
pyplot.xlabel('n')
```

![](mdImages/2021-02-17-12-52-27.png)

</details>

---

## Array

### Dynamic Array

<details>

  <summary>Own Implementation</summary>

```py
import ctypes

class DynamicArray(object):

  def __init__(self):
      self.index = 0
      self.size = 0
      self.capacity = 1
      self.array = self.make_array(self.capacity)

  def __len__(self):
      return self.size

  def __getitem__(self, k):
      if not 0 <= k < self.size:
          raise IndexError('Out of bounds!')
      return self.array[k]

  def append(self, element):
      if self.size == self.capacity:
          self._resize(2 * self.capacity)

      self.array[self.size] = element
      self.size += 1

  def _resize(self, new_cap):
      temp_array = self.make_array(new_cap)
      for k in range(self.size):
          temp_array[k] = self.array[k]
      self.array = temp_array
      self.capacity = new_cap

  def make_array(self, new_cap):
      return (new_cap * ctypes.py_object)()

  def __iter__(self):
      return self

  def __next__(self):
      if self.index > self.size:
          raise StopIteration
      else:
          item = self.array[self.index]
          self.index += 1
          return item
```

</details>

### Amortized Analysis

<details>

  <summary>Cost of successive appends on a list</summary>

![](mdImages/2021-02-17-12-58-56.png)

</details>

### Problems

<details>

  <summary>Anagram Check</summary>

#### Problem

Given two strings, check to see if they are anagrams. An anagram is when the two strings can be written using the exact same letters (so you can just rearrange the letters to get a different phrase or word).

For example:

"public relations" is an anagram of "crap built on lies."

"clint eastwood" is an anagram of "old west action"

Note: Ignore spaces and capitalization. So "d go" is an anagram of "God" and "dog" and "o d g".

#### Solution

**Solution by Sorting**

```py
  def sol_anagram_sorting(string1: str, string2: str) -> bool:
    string1 = string1.replace(' ', '').lower()
    string2 = string2.replace(' ', '').lower()

    return sorted(string1) == sorted(string2)
```

**By using Dictionary**

```py
import string
def sol_anagram(string1: str, string2: str) -> bool:
    string1 = string1.replace(' ', '').lower()
    string2 = string2.replace(' ', '').lower()

    if len(string1) != len(string2): return False

    dict1 = dict.fromkeys(string.ascii_lowercase, 0)
    dict2 = dict.fromkeys(string.ascii_lowercase, 0)
    for char in string1: dict1[char] += 1
    for char in string2: dict2[char] += 1

    for (key, value) in dict1.items():
        if dict2[key] != value: return False

    return True
```

**Output**

![](mdImages/2021-02-18-15-57-21.png)

</details>

<details>

  <summary>Array Pair Sum</summary>

#### Problem

Given an integer array, output all the unique pairs that sum up to a specific value k.

So the input:

pair_sum([1,3,2,2],4)

would return 2 pairs:

(1,3)
(2,2)

NOTE: FOR TESTING PURPOSES CHANGE YOUR FUNCTION SO IT OUTPUTS THE NUMBER OF PAIRS

#### Solution

**Using Set Method**

```py
def sol_pair_sum_set(array: 'list[int]', value: int) -> int:
    if len(array) < 2: return -1
    seen = set()
    output = set()

    for num in array:
        target = value - num
        if target not in seen:
            seen.add(num)
        else:
            output.add((min(target, num), max(target, num)))
    return len(output)
```

**Scratch Method**

```py
def sol_pair_sum(array: [int], value: int) -> int:
    array.sort()
    sindex = 0
    eindex = len(array) - 1
    count = 0
    while sindex < eindex:
        current_value = array[sindex] + array[eindex]
        if current_value > value:
            eindex -= 1
        elif current_value < value:
            sindex += 1
        else:
            count += 1
            sindex += 1
            eindex -= 1
    return count
```

**Output**

![](mdImages/2021-02-18-16-04-05.png)

</details>

<details>

  <summary>Find the Missing Element</summary>

#### Problem

Consider an array of non-negative integers. A second array is formed by shuffling the elements of the first array and deleting a random element. Given these two arrays, find which element is missing in the second array.

Here is an example input, the first array is shuffled and the number 5 is removed to construct the second array.

Input:

finder([1,2,3,4,5,6,7],[3,7,2,1,4,6])

Output:

5 is the missing number

#### Solution

**Using Sorting**

```py
def sol_find_missing_sort(array1: 'list[int]', array2: 'list[int]') -> int:
    array1.sort()
    array2.sort()

    for num1, num2 in zip(array1, array2):
        if num1 != num2:
            return num1
    return num1[-1]
```

**Using Hashing**

```py
def sol_find_missing_hash(array1: 'list[int]', array2: 'list[int]') -> int:
    count = dict()
    for item in array2:
        if item not in count:
            count[item] = 1
        else:
            count[item] += 1

    for item in array1:
        if item not in count:
            return item

        elif count[item] == 0:
            return item

        else:
            count[item] -= 1
```

**Ouput**

![](mdImages/2021-02-18-16-06-37.png)

</details>

<details>

  <summary>Largest Continuous Sum</summary>

#### Problem

Given an array of integers (positive and negative) find the largest continuous sum.

#### Solution

**Kadane's Algorithm**

```py
def large_cont_sum(array: 'list[int]') -> int:
    if len(array) == 0: return 0

    max_sum = current_max = array[0]

    for num in array[1:]:
        current_max = max(current_max + num, num)
        max_sum = current_max if current_max > max_sum else max_sum

    return max_sum
```

**Output**

![](mdImages/2021-02-18-16-10-57.png)

</details>

<details>

  <summary>Sentence Reversal</summary>

#### Problem

Given a string of words, reverse all the words. For example:

Given:

'This is the best'

Return:

'best the is This'

As part of this exercise you should remove all leading and trailing whitespace. So that inputs such as:

' space here' and 'space here '

both become:

'here space'

#### Solution

**Scratch Method**

```py
def sol_rev_word_scratch(string: str) -> str:
    words = []
    i = 0
    size = len(string)
    space = ' '
    while i < size:
        if string[i] is not space:
            start_index = i
            while i < size and string[i] is not space:
                i += 1
            words.append(string[start_index : i])
        i += 1

    start = 0
    end = len(words) - 1
    while start < end:
        temp = words[start]
        words[start] = words[end]
        words[end] = temp
        start += 1
        end -= 1

    result = ''
    for i in range(len(words)):
        result += (words[i] + ' ') if i != len(words) - 1 else (words[i])
    return result
```

**Pythonic Way**

```py
def sol_rev_word(string: str) -> str:
    return ' '.join(reversed(string.strip(' ').split()))
```

**Output**

![](mdImages/2021-02-18-16-13-01.png)

</details>

<details>

  <summary>String Compression</summary>

#### Problem

Given a string in the form 'AAAABBBBCCCCCDDEEEE' compress it to become 'A4B4C5D2E4'. For this problem, you can falsely "compress" strings of single or double letters. For instance, it is okay for 'AAB' to return 'A2B1' even though this technically takes more space.

The function should also be case sensitive, so that a string 'AAAaaa' returns 'A3a3'.

#### Solution

```py
def sol_compress(string: str) -> str:
    size = len(string)

    if size == 0: return ''

    if size == 1: return string[0] + '1'

    result = ''
    count = 1
    i = 1

    while i < size:
        if string[i] == string[i - 1]:
            count += 1
        else:
            result += (string[i-1] + str(count))
            count = 1
        i += 1

    result += (string[i-1] + str(count))
    return result
```

**Output**

![](mdImages/2021-02-18-16-15-13.png)

</details>

<details>

  <summary>Unique Characters in String</summary>

#### Problem

Given a string,determine if it is compreised of all unique characters. For example, the string 'abcde' has all unique characters and should return True. The string 'aabcde' contains duplicate characters and should return false.

#### Solution

**One liner Set Method**

```py
def sol_uni_char(string: str) -> bool:
    return len(set(string)) == len(string)
```

**Lookup Scratch**

```py
def sol_uni_char_look(string: str) -> bool:
    str_set = set()
    for char in string:
        if char in str_set:
            return False
        else:
            str_set.add(char)
    return True
```

**Output**

![](mdImages/2021-02-18-16-17-15.png)

</details>

## Stack, Queue, Deque

### What is a Stack

![](mdImages/2021-02-18-17-21-40.png)

> LIFO Principle

<details>

  <summary>Overview</summary>

A stack is an ordered collection of items where the addition of new items and the removal of existing items always takes place at the same end. This end is commonly referred to as the “top.” The end opposite the top is known as the “base.”

The base of the stack is significant since items stored in the stack that are closer to the base represent those that have been in the stack the longest. The most recently added item is the one that is in position to be removed first.

This ordering principle is sometimes called LIFO, last-in first-out. It provides an ordering based on length of time in the collection. Newer items are near the top, while older items are near the base.

For example, consider the figure below:

![](https://upload.wikimedia.org/wikipedia/commons/b/b4/Lifo_stack.png)

Note how the first items "pushed" to the stack begin at the base, and as items are "popped" out. Stacks are fundamentally important, as they can be used to reverse the order of items. The order of insertion is the reverse of the order of removal.

Considering this reversal property, you can perhaps think of examples of stacks that occur as you use your computer. For example, every web browser has a Back button. As you navigate from web page to web page, those pages are placed on a stack (actually it is the URLs that are going on the stack). The current page that you are viewing is on the top and the first page you looked at is at the base. If you click on the Back button, you begin to move in reverse order through the pages.

</details>

<details>

  <summary>Stack Implementation</summary>

Before we implement our own Stack class, let's review the properties and methods of a Stack.

The stack abstract data type is defined by the following structure and operations. A stack is structured, as described above, as an ordered collection of items where items are added to and removed from the end called the “top.” Stacks are ordered LIFO. The stack operations are given below.

- Stack() creates a new stack that is empty. It needs no parameters and returns an empty stack.
- push(item) adds a new item to the top of the stack. It needs the item and returns nothing.
- pop() removes the top item from the stack. It needs no parameters and returns the item. The stack is modified.
- peek() returns the top item from the stack but does not remove it. It needs no parameters. The stack is not modified.
- isEmpty() tests to see whether the stack is empty. It needs no parameters and returns a boolean value.
- size() returns the number of items on the stack. It needs no parameters and returns an integer.

```py
class Stack(object):
  def __init__(self):
      self.items = []

  def is_empty(self):
      return len(self) == 0

  def push(self, item):
      self.items.append(item)

  def pop(self):
      return self.items.pop()

  def peek(self):
      return self.items[-1]

  def size(self):
      return len(self.items)

  def __len__(self):
      return len(self.items)

  def __iter__(self):
      self.index = 0
      return self

  def __next__(self):
      if self.index < 0 or self.index > len(self):
          raise StopIteration
      else:
          item =  self.items[-1 * (index + 1)]
          self.index += 1
          return item

  def __getitem__(self, index):
      return self.items[index]

  def __repr__(self):
      return ', '.join([str(item) for item in self.items])
```

</details>

### What is a Queue

![](mdImages/2021-02-18-17-35-59.png)

> FIFO Principle

<details>

  <summary>Overview</summary>

A queue is an ordered collection of items where the addition of new items happens at one end, called the “rear,” and the removal of existing items occurs at the other end, commonly called the “front.” As an element enters the queue it starts at the rear and makes its way toward the front, waiting until that time when it is the next element to be removed.

The most recently added item in the queue must wait at the end of the collection. The item that has been in the collection the longest is at the front. This ordering principle is sometimes called FIFO, first-in first-out. It is also known as “first-come first-served.”

The simplest example of a queue is the typical line that we all participate in from time to time. We wait in a line for a movie, we wait in the check-out line at a grocery store, and we wait in the cafeteria line. The first person in that line is also the first person to get serviced/helped.

Let's see a diagram which shows this and compares it to the Stack Data Structure:

![](https://netmatze.files.wordpress.com/2014/08/queue.png)

Note how we have two terms here, Enqueue and Dequeue. The enqueue term describes when we add a new item to the rear of the queue. The dequeue term describes removing the front item from the queue.

</details>

<details>

  <summary>Queue Implementation</summary>

Before we begin implementing our own queue, let's review the attribute and methods it will have:

- Queue() creates a new queue that is empty. It needs no parameters and returns an empty queue.
- enqueue(item) adds a new item to the rear of the queue. It needs the item and returns nothing.
- dequeue() removes the front item from the queue. It needs no parameters and returns the item. The queue is modified.
- isEmpty() tests to see whether the queue is empty. It needs no parameters and returns a boolean value.
- size() returns the number of items in the queue. It needs no parameters and returns an integer.

```py
class Queue(object):

    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < 0 or self.index > len(self.items) - 1:
            raise StopIteration
        else:
            item = self.items[self.index]
            self.index += 1
            return item

    def __repr__(self):
        return 'Back => ' + ', '.join([str(item) for item in self]) + ' => Front'
```

</details>

### What is a Deque

![](mdImages/2021-02-18-17-58-28.png)

> Insertion and Removal on either end

<details>

  <summary>Overview</summary>

A deque, also known as a double-ended queue, is an ordered collection of items similar to the queue. It has two ends, a front and a rear, and the items remain positioned in the collection. What makes a deque different is the unrestrictive nature of adding and removing items. New items can be added at either the front or the rear. Likewise, existing items can be removed from either end. In a sense, this hybrid linear structure provides all the capabilities of stacks and queues in a single data structure.

It is important to note that even though the deque can assume many of the characteristics of stacks and queues, it does not require the LIFO and FIFO orderings that are enforced by those data structures. It is up to you to make consistent use of the addition and removal operations.

Let's see an Image to visualize the Deque Data Structure:

![](http://www.codeproject.com/KB/recipes/669131/deque.png)

Note how we can both add and remove from the front and the back of the Deque.

</details>

<details>

  <summary>Deques Implementation</summary>

```py
class Deque(object):

    def __init__(self):
        self.items = []

    def add_front(self, item):
        self.items.append(item)

    def add_rear(self, item):
        self.items.insert(0, item)

    def remove_front(self):
        self.items.pop()

    def remove_rear(self):
        self.items.pop(0)

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < 0 or self.index > len(self.items) - 1:
            raise StopIteration
        else:
            item = self.items[self.index]
            self.index += 1
            return item

    def __repr__(self):
        return 'Back => ' + ', '.join([str(item) for item in self]) + ' => Front'
```

</details>

### Problems

<details>

  <summary>Balanced Parentheses Check</summary>

#### Problem

Given a string of opening and closing parentheses, check whether it’s balanced. We have 3 types of parentheses: round brackets: (), square brackets: [], and curly brackets: {}. Assume that the string doesn’t contain any other character than these, no spaces words or numbers. As a reminder, balanced parentheses require every opening parenthesis to be closed in the reverse order opened. For example ‘([])’ is balanced but ‘([)]’ is not.

You can assume the input string has no spaces.

#### Solution

This is a very common interview question and is one of the main ways to check your knowledge of using Stacks! We will start our solution logic as such:

First we will scan the string from left to right, and every time we see an opening parenthesis we push it to a stack, because we want the last opening parenthesis to be closed first. (Remember the FILO structure of a stack!)

Then, when we see a closing parenthesis we check whether the last opened one is the corresponding closing match, by popping an element from the stack. If it’s a valid match, then we proceed forward, if not return false.

Or if the stack is empty we also return false, because there’s no opening parenthesis associated with this closing one. In the end, we also check whether the stack is empty. If so, we return true, otherwise return false because there were some opened parenthesis that were not closed.

**Lengthy Method**

```py
from collections import deque

def sol_balance_check(string: str) -> bool:
    if len(string) % 2 != 0:
        return False

    open_parens = ['(', '{', '[']
    close_parens = [')', '}', ']']
    paren_stack = deque()

    for char in string:
        if char in open_parens:
            paren_stack.append(char)
        elif char in close_parens:
            open_index = open_parens.index(paren_stack[-1])
            close_index = close_parens.index(char)
            if open_index == close_index:
                paren_stack.pop()
            else:
                return False

    if len(paren_stack) != 0:
        return False
    return True
```

**Simple Method**

```py
from collections import deque
def sol_balance_check_easy(string: str) -> bool:
    if len(string) % 2 != 0:
        return False

    open_parens = set(['(', '[', '{'])
    close_parens = set([')', ']', '}'])
    pair_parens = set([('(', ')'), ('{', '}'), ('[', ']')])

    stack = deque()

    for char in string:
        if char in open_parens:
            stack.append(char)
        elif char in close_parens:
            if (stack.pop(), char) not in pair_parens:
                return False

    return len(stack) == 0
```

**Output**

![](mdImages/2021-02-18-20-37-19.png)

</details>

<details>

  <summary>Implement a Queue - Using Two Stacks</summary>

#### Problem

Given the Stack class below, implement a Queue class using two stacks! Note, this is a "classic" interview problem. Use a Python list data structure as your Stack.

#### Solution

The key insight is that a stack reverses order (while a queue doesn't). A sequence of elements pushed on a stack comes back in reversed order when popped. Consequently, two stacks chained together will return elements in the same order, since reversed order reversed again is original order.

We use an in-stack that we fill when an element is enqueued and the dequeue operation takes elements from an out-stack. If the out-stack is empty we pop all elements from the in-stack and push them onto the out-stack.

```py
from collections import deque

class Dequeu(object):

    def __init__(self):
        self.in_stack = deque()
        self.out_stack = deque()

    def enqueue(self, item):
        self.in_stack.append(item)

    def deque(self):
        if len(self.out_stack) == 0:
            self.transfer()
        return self.out_stack.pop()

    def transfer(self):
        while (self.in_stack) is not 0:
            self.out_stack.append(self.in_stack.pop())
```

**Output**

![](mdImages/2021-02-18-20-49-12.png)

</details>

## Linked List

### What is Linked List

![](mdImages/2021-02-18-20-55-23.png)

<details>

  <summary>Overview</summary>

Remember, in a singly linked list, we have an ordered list of items as individual Nodes that have pointers to other Nodes.

In a Linked List the first node is called the head and the last node is called the tail. Let's discuss the pros and cons of Linked Lists:

#### Pros

Linked Lists have constant-time insertions and deletions in any position, in comparison, arrays require O(n) time to do the same thing.

Linked lists can continue to expand without having to specify their size ahead of time (remember our lectures on Array sizing form the Array Sequence section of the course!)

#### Cons

To access an element in a linked list, you need to take O(k) time to go from the head of the list to the kth element. In contrast, arrays have constant time operations to access elements in an array.

</details>

<details>

  <summary>Implementation of a Node</summary>

```py
class Node(object):

  def __init__(self, value, next = None):
      self.value = value
      self.next = None

  def __repr__(self):
      result = f'(Node: {self.value}) -> '
      if self.next is None:
          result += 'None'
      else:
          result += str(self.next)
      return result
```

</details>

### What is Doubly Linked List

![](mdImages/2021-02-18-21-17-55.png)

<details>

  <summary>Implementation of Doubly Linked List</summary>

```py
 class DoublyNode(object):

    def __init__(self, value, prev = None, next = None):
        self.value = value
        self.next = next
        self.prev = prev
    def __repr__(self):
        result = f'Node: {self.value} <=> '
        if self.next is None:
            result += 'None'
        else:
            result += str(self.next)
        return result
```

</details>

### Problems

<details>

  <summary>Singly Linked List Cycle Check</summary>

#### Problem

Given a singly linked list, write a function which takes in the first node in a singly linked list and returns a boolean indicating if the linked list contains a "cycle".

A cycle is when a node's next point actually points back to a previous node in the list. This is also sometimes known as a circularly linked list.

#### Solution

To solve this problem we will have two markers traversing through the list. marker1 and marker2. We will have both makers begin at the first node of the list and traverse through the linked list. However the second marker, marker2, will move two nodes ahead for every one node that marker1 moves.

By this logic we can imagine that the markers are "racing" through the linked list, with marker2 moving faster. If the linked list has a cycle and is circularly connected we will have the analogy of a track, in this case the marker2 will eventually be "lapping" the marker1 and they will equal each other.

If the linked list has no cycle, then marker2 should be able to continue on until the very end, never equaling the first marker.

```py
def cycle_check(node: Node) -> bool:
    marker1 = node
    marker2 = node

    while marker1 != None and marker2.next != None:
        marker1 = marker1.next
        marker2 = marker2.next.next

        if marker1 == marker2:
            return True
    return False
```

</details>

<details>

  <summary>Reverse a Linked List</summary>

#### Problem

Write a function to reverse a Linked List in place. The function will take in the head of the list as input and return the new head of the list.

You are given the example Linked List Node class:

#### Solution

Since we want to do this in place we want to make the function operate in O(1) space, meaning we don't want to create a new list, so we will simply use the current nodes! Time wise, we can perform the reversal in O(n) time.

We can reverse the list by changing the next pointer of each node. Each node's next pointer should point to the previous node.

In one pass from head to tail of our input list, we will point each node's next pointer to the previous element.

```py
def reverse_linked_list(head: Node) -> None:
    current_node = head
    prev_node = None
    next_node = None

    # Until we have gone through the end of the list
    while current_node is not None:
        # Copy the current_node's next to next_node
        next_node = current_node.next

        # Reverse the pointer of the next_node
        current_node.next = prev_node

        # Go one forward in the list
        prev_node = current_node
        current_node = next_node

    return prev_node
```

</details>

<details>

  <summary>Linked List Nth to Last Node</summary>

#### Problem

Write a function that takes a head node and an integer value n and then returns the nth to last node in the linked list. For example, given:

#### Solution

One approach to this problem is this:

Imagine you have a bunch of nodes and a "block" which is n-nodes wide. We could walk this "block" all the way down the list, and once the front of the block reached the end, then the other end of the block would be a the Nth node!

So to implement this "block" we would just have two pointers a left and right pair of pointers. Let's mark out the steps we will need to take:

- Walk one pointer n nodes from the head, this will be the right_point
- Put the other pointer at the head, this will be the left_point
- Walk/traverse the block (both pointers) towards the tail, one node at a time, keeping a distance n between them.
- Once the right_point has hit the tail, we know that the left point is at the target.

Let's see the code for this!

```py
def nth_to_last_node(n: int, head: Node) -> Node:
    r_node = l_node = head
    for i in range(n - 1):
        if r_node.next == None:
            raise LookupError('Error: n > size of linked list')
        else:
            r_node = r_node.next

    while r_node.next != None:
        r_node = r_node.next
        l_node = l_node.next

    return l_node
```

</details>

## Recursion

![](mdImages/2021-02-19-11-20-15.png)

### Problems

<details>

  <summary>Problem 1</summary>

#### Problem

Write a recursive function which takes an integer and computes the cumulative sum of 0 to that integer

For example, if n=4 , return 4+3+2+1+0, which is 10.

This problem is very similar to the factorial problem presented during the introduction to recursion. Remember, always think of what the base case will look like. In this case, we have a base case of n =0 (Note, you could have also designed the cut off to be 1).

In this case, we have: n + (n-1) + (n-2) + .... + 0

#### Solution

```py
def rec_sum(n):
    if n == 0:
        return 0
    else:
        return n + rec_sum(n - 1)

# Testing
rec_sum(4)
```

</details>

<details>

  <summary>Problem 2</summary>

#### Problem

Given an integer, create a function which returns the sum of all the individual digits in that integer. For example: if n = 4321, return 4+3+2+1

#### Solution

```py
def sum_func(n):
    if n == 0:
        return 0
    return n % 10 + sum_func(n // 10)

# Testing
sum_func(4321)
```

</details>

<details>

  <summary>Problem 3</summary>

#### Problem

Note, this is a more advanced problem than the previous two! It aso has a lot of variation possibilities and we're ignoring strict requirements here.

Create a function called word_split() which takes in a string phrase and a set list_of_words. The function will then determine if it is possible to split the string in a way in which words can be made from the list of words. You can assume the phrase will only contain words found in the dictionary if it is completely splittable.

#### Solution

```py
def word_split(string: str, words: 'list[str]') -> bool:
    if len(string) == 0:
        return True
    for word in words:
        if string[:len(word)] == word:
            return word_split(string[ len(word): ], words)
    return False

# Testing
word_split('themanran',['the','ran','man'])
word_split('ilovedogsJohn',['i','am','a','dogs','lover','love','John'])
word_split('themanran',['clown','ran','man'])
```

</details>

### Memoization

We will discuss memoization and dynamic programming. For your homework assignment, read the Wikipedia article on Memoization, before continuing on with this lecture!

Memoization effectively refers to remembering ("memoization" -> "memorandum" -> to be remembered) results of method calls based on the method inputs and then returning the remembered result rather than computing the result again. You can think of it as a cache for method results. We'll use this in some of the interview problems as improved versions of a purely recursive solution.

A simple example for computing factorials using memoization in Python would be something like this:

```py
# Create cache for known results
factorial_memo = {}

def factorial(k):

    if k < 2:
        return 1

    if not k in factorial_memo:
        factorial_memo[k] = k * factorial(k-1)

    return factorial_memo[k]
```

Note how we are now using a dictionary to store previous results of the factorial function! We are now able to increase the efficiency of this function by remembering old results!

Keep this in mind when working on the Coin Change Problem and the Fibonacci Sequence Problem.

We can also encapsulate the memoization process into a class:

```py
class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]
```

Then all we would have to do is:

```py
def factorial(k):

    if k < 2:
        return 1

    return k * factorial(k - 1)

factorial = Memoize(factorial)
```

Try comparing the run times of the memoization versions of functions versus the normal recursive solutions!

### Problems

<details>

  <summary>String Reversal</summary>

#### Problem

This interview question requires you to reverse a string using recursion. Make sure to think of the base case here.

Again, make sure you use recursion to accomplish this. Do not slice (e.g. string[::-1]) or use iteration, there muse be a recursive call for the function.

#### Solution

In order to reverse a string using recursion we need to consider what a base and recursive case would look like. Here we've set a base case to be when the length of the string we are passing through the function is length less than or equal to 1.

During the recursive case we grab the first letter and add it on to the recursive call.

```py
def reverse(string: str) -> str:
    if len(string) is 0: return ''
    return reverse(string[1:]) + string[0]

# Testing
reverse('hello world')
```

</details>

<details>

  <summary>String Permutation</summary>

#### Problem

Given a string, write a function that uses recursion to output a list of all the possible permutations of that string.

For example, given s='abc' the function should return ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']

Note: If a character is repeated, treat each occurence as distinct, for example an input of 'xxx' would return a list with 6 "versions" of 'xxx'

#### Solution

```py
def permute(string: str) -> 'list[str]':
    output = []

    # Base Case
    if len(string) == 1:
        return [string]

    # for every letter in string
    for i, let in enumerate(string):
        for perm in permute(string[:i] + string[i+1:]):
            output += [let + perm]

    return output
```

**Conclusion**

There were two main takeaways from tackling this problem:

- Every time we put a new letter in position i, we then had to find all the possible combinations at position i+1 – this was the recursive call that we made. How do we know when to save a string? When we are at a position i that is greater than the number of letters in the input string, then we know that we have found one valid permutation of the string and then we can add it to the list and return to changing letters at positions less than i. This was our base case – remember that we always must have a recursive case and a base case when using recursion!
- Another big part of this problem was figuring out which letters we can put in a given position. Using our sample string “abc”, lets say that we are going through all the permutations where the first letter is "c”. Then, it should be clear that the letter in the 2nd and 3rd position can only be either “a” or “b”, because “a” is already used. As part of our algorithm, we have to know which letters can be used in a given position – because we can’t reuse the letters that were used in the earlier positions.

</details>

<details>

  <summary>Fibonacci Different Ways!</summary>

#### Problem

Implement a Fibonnaci Sequence in three different ways:

Recursively
Dynamically (Using Memoization to store results)
Iteratively
Remember that a fibonacci sequence: 0,1,1,2,3,5,8,13,21,... starts off with a base case checking to see if n = 0 or 1, then it returns 1.

Else it returns fib(n-1)+fib(n+2).

#### Solution

**Recursively**

The recursive solution is exponential time Big-O , with O(2^n). However, its a very simple and basic implementation to consider:

```py
def fib_rec(num: int) -> int:
    if (num < 2): return num
    return fib_rec(num - 1) + fib_rec(num - 2)

# Testing
fib_rec(10)
```

**Dynamically**

In the form it is implemented here, the cache is set beforehand and is based on the desired n number of the Fibonacci Sequence. Note how we check it the cache[n] != None, meaning we have a check to know wether or not to keep setting the cache (and more importantly keep cache of old results!)

```py
def fib_dyn(num: int, memo = dict()) -> int:
    if num < 2:
        return num
    if num in memo:
        return memo.get(num)

    ans = fib_dyn(num - 1, memo) + fib_dyn(num - 2, memo)
    memo[num] = ans

    return ans

# Testing
fib_dyn(10)
```

**Iteratively**

```py
def fib_iter(num: int) -> int:
    prev = 0
    cur = 1

    for i in range(num):
        prev, cur = cur, prev + cur

    return prev
```

</details>

<details>

  <summary>Coin Change Problem</summary>

#### Problem

This problem has multiple solutions and is a classic problem in showing issues with basic recursion. There are better solutions involving memoization and simple iterative solutions.If you are having trouble with this problem (or it seems to be taking a long time to run in some cases) check out the Solution Notebook and fully read the conclusion link for a detailed description of the various ways to solve this problem!

This is a classic recursion problem: Given a target amount n and a list (array) of distinct coin values, what's the fewest coins needed to make the change amount.

For example:

If n = 10 and coins = [1,5,10]. Then there are 4 possible ways to make change:

1+1+1+1+1+1+1+1+1+1

5 + 1+1+1+1+1

5+5

10

With 1 coin being the minimum amount.

#### Solution

**Recursive Solution**

```py
def rec_coin(target: int, coins: 'list[int]') -> int:
    if target == 0:
        return 0

    if target < 0:
        return float('inf')

    min_coins = float('inf')

    for coin in coins:
        num_coins = 1 + rec_coin(target - coin, coins)
        if num_coins < min_coins:
            min_coins = num_coins

    return min_coins
```

**Memoization Solution**

```py
def rec_coin_dynamic(target: int, coins: 'list[int]', memo = dict()) -> int:
    if target == 0:
        return 0

    if target < 0:
        return float('inf')

    if target in memo:
        return memo.get(target)

    min_coins = float('inf')

    for coin in coins:
        num_coins = rec_coin_dynamic(target - coin, coins, memo) + 1

        if num_coins < min_coins:
            min_coins = num_coins

    memo[target] = min_coins
    return min_coins
```

</details>

## Dynamic Programming

### Memoization Recipe

<details>

  <summary>Make it work</summary>

- visualize the problem
- implement the tree using recursion
- test it

</details>

<details>

  <summary>Make it efficient</summary>

- add a memo object
- add a base case to return memo values
- store return values into the memo

</details>

### Fibonacci Problem

<details>

  <summary>Solution and Complexity</summary>

![](mdImages/2021-02-19-20-52-50.png)

**Fibo Tree**

![](mdImages/2021-02-19-21-23-24.png)

```py
def fib_rec(num: int, memo = dict()) -> int:
    if num < 2:
        return num
    if num in memo:
        return memo.get(num)

    ans = fib_rec(num - 1, memo) + fib_rec(num - 2, memo)
    memo[num] = ans
    return ans
```

</details>

### Grid Traveler Problem

<details>

  <summary>Solution and Complexity</summary>

![](mdImages/2021-02-19-20-59-38.png)

**Grid Traveler Tree**

![](mdImages/2021-02-19-21-21-53.png)

```py
  def grid_traveler_rec(rows: int, cols: int, memo = dict()) -> int:
    if rows is 0 or cols is 0:
        return 0
    if rows is 1 and cols is 1:
        return 1

    if f'{rows},{cols}' in memo:
        return memo[f'{rows},{cols}']

    ans = grid_traveler_rec(rows - 1, cols, memo) + grid_traveler_rec(rows, cols - 1, memo)
    memo[f'{rows},{cols}'] = ans
    return ans
```

</details>

### Can Sum DP Type

<details>

  <summary>Procedure</summary>

- Write a function **can_sum(target_sum, numbers)** that takes in a **target_sum** and an array of numbers as arguments
- The function should return a boolean indicating whether it is possible to generate the **target_sum** using numbers from the array
- You may use an element of the array as many times as needed
- You may assume that all input numbers are non-negative

</details>

<details>

  <summary>Solution and Complexity</summary>

![](mdImages/2021-02-19-21-17-15.png)

**Can Sum Tree**

![](mdImages/2021-02-19-21-20-19.png)

```py
def can_sum_rec(target: int, nums: int, memo = dict()) -> bool:
    if target == 0:
        return True

    if target < 0:
        return False

    if target in memo:
        return memo.get(target)

    for num in nums:
        if can_sum_rec(target - num, nums, memo):
            memo[target] = True
            return True

    memo[target] = False
    return False
```

</details>

### How Sum DP Type

<details>

  <summary>Procedure</summary>

- Write a function **how_sum(target_sum, numbers)** that takes in a target_sum and an array of numbers as arguments
- The function should return an array containing any combination of elements that add up to exactly that **target_sum**. If there is no combination that adds up to the **target_sum**, then return null
- If there are multiple combinations possible, you may return any single one

</details>

<details>

  <summary>Solution and Complexity</summary>

![](mdImages/2021-02-19-21-35-50.png)

**How Sum Tree**

![](mdImages/2021-02-19-21-36-20.png)

```py
def how_sum_rec(target: int, nums: 'list[int]', memo = dict()) -> 'list[int]':
    if target == 0:
        return []

    if target < 0:
        return None

    if target in memo:
        return memo.get(target)

    for num in nums:
        current = how_sum_rec(target - num, nums, memo)
        if current is not None:
            current.append(num)
            memo[target] = current
            return current
    memo[target] = None
    return None
```

</details>

### Best Sum DP Type

<details>

  <summary>Procedure</summary>

- Write a function **best_sum(target_sum, numbers)** that takes in a **target_sum** and an array of numbers an an argument
- The function should return an array containing the **shortest** combination of numbers that add up to exactly the **target_sum**
- If there is a tie for the shortest combination, you may return any one of the shortest

</details>

<details>

  <summary>Solution and Complexity</summary>

![](mdImages/2021-02-19-21-46-44.png)

**Best Sum Tree**

![](mdImages/2021-02-19-21-47-21.png)

```py
def best_sum_rec(target: int, nums: 'list[int]', memo = dict()):
    if target == 0:
        return []

    if target < 0:
        return None

    if target in memo:
        return memo.get(target)

    min_array = None
    for num in nums:
        current_array = best_sum_rec(target - num, nums, memo)
        if current_array is not None:
            current_array.append(num)
            if min_array is None:
                min_array = current_array
            elif len(current_array) < len(min_array):
                min_array = current_array
    memo[target] = min_array
    return min_array
```

</details>

### In a Nutshell

- **can_sum => Decision Problem**
  _(can you do it? yes/no)_

  > m = target_sum
  > n = array length

  | Brute Force   | Memoized        |
  | ------------- | --------------- |
  | O($n^m$) time | O($m * n$) time |
  | O($m$) space  | O($m$) space    |

- **how_sum => Combinatoric Problem**
  _(how will you do it?)_

  > m = target_sum
  > n = array length

  | Brute Force       | Memoized          |
  | ----------------- | ----------------- |
  | O($n^m * m$) time | O($n * m^2$) time |
  | O($m$) space      | O($m^2$) space    |

- **best_sum => Optimization Problem**
  _(what is the best way to do it?)_

  > m = target_sum
  > n = array length

  | Brute Force       | Memoized          |
  | ----------------- | ----------------- |
  | O($n^m * m$) time | O($n * m^2$) time |
  | O($m^2$) space    | O($m^2$) space    |

### Can Construct DP Type
