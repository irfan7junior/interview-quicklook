# Interview Quicklook

---

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
