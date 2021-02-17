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

  <summary>Cost of successive append on list</summary>

![](mdImages/2021-02-17-12-58-56.png)

</details>

---
