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

## </details>

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
