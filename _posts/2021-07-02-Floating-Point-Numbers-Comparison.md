---
layout: post
title: "Does 0.1 + 0.2 equal 0.3?"
tagline: 
author: "Wangbo Zheng"
---

It seems like a simple question, that a primary school student can easily answer. But if you try the following code in Python, you will get a False.

```python
0.1 + 0.2 == 0.3
```

> False

You can also try the classic one:

```python
0.1 + 0.1 + 0.1 == 0.3
```

> False

The output is still false. "This is not a bug in Python, and it is not a bug in your code either. You’ll see the same kind of thing in all languages that support your hardware’s floating-point arithmetic." The reason is that in computer hardware all floating-point numbers are represented as binary fractions. Unfortunately, most decimal fractions cannot be represented exactly as binary fractions. "A consequence is that, in general, the decimal floating-point numbers you enter are only approximated by the binary floating-point numbers actually stored in the machine." In the case of 0.1, the binary fraction is 3602879701896397 / 2 ** 55 which is extremely close to but not exactly equal to the true value of 1/10.

```python
0.1 == 3602879701896397 / 2 ** 55
```

> True

A helpful function from the math module is the ```math.fsum()```. It helps decrease loss-of-precision during summation. It makes the outcome of the following scenario different.

```python
sum([.1]*10) == 1
```

> False

```python
import math
math.fsum([0.1]*10) == 1
```

> True

However, it is still not helpful for our 0.3 example.

```python
math.fsum([0.1, 0.1, 0.1]) == 0.3
```

> False

So what is the best way to compare floating-point numbers? Since python 3.5 a function in math module ```isclose()``` was added to tell whether two values are approximately equal or “close” to each other. You decide the relative tolerance which is the maximum allowed difference between ```isclose()``` arguments, relative to the larger absolute value.

```python
math.isclose(0.1+0.2, 0.3, abs_tol=1e-5)
```

> True

There is also an ```isclose()``` function in Numpy, which allows you to compare the elements in an array. 

```python
import numpy as np
np.isclose([0.1+0.1, 0.1+0.1+0.1], [0.2, 0.3])
```

> array([ True,  True])

A very important application scenario with floating-point numbers comparison in the test process. The ```approx()``` is the correct function from the pytest library to assert that two numbers are equal to each other within some tolerance.

```python
from pytest import approx
0.1 + 0.2 == approx(0.3)
```

> True

### References

1. [Floating Point Arithmetic: Issues and Limitations ](https://docs.python.org/3/tutorial/floatingpoint.html)

2. [What’s New In Python 3.5](https://docs.python.org/3/whatsnew/3.5.html#pep-485-a-function-for-testing-approximate-equality)

3. [math fsum](https://docs.python.org/3/library/math.html#math.fsum)

4. [numpy.isclose](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)

5. [What is the best way to compare floats for almost-equality in Python? - Stack Overflow](https://stackoverflow.com/questions/5595425/what-is-the-best-way-to-compare-floats-for-almost-equality-in-python)

6. [pytest documentation approx](https://docs.pytest.org/en/latest/reference/reference.html?highlight=approx#pytest.approx)

   

   

