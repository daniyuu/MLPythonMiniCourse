import numpy
import pandas

# numpy
myarray = numpy.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']
mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
print(mydataframe)

a = numpy.zeros((2, 2))
print(a)

b = numpy.ones((1, 2))
print(b)

# c = numpy.full((2, 2), 7)
# print(c)

d = numpy.eye(2)
print(d)

e = numpy.random.random((2, 2))
print(e)

# Array indexing
a = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
b = a[:2, 1:3]

print(b)
print(a)
# A slice of an array is a view into the same data, so modifying it will modify the original array
print(a[0, 1])
b[0, 0] = 77
print(a[0, 1])

a = numpy.array([[1, 2], [3, 4], [5, 6]])

print(a[[0, 1, 2], [0, 1, 0]])

# The above example of integer array indexing is equivalent to this
print(numpy.array([a[0, 0], a[1, 1], a[2, 0]]))

# useful index
a = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(a)
b = numpy.array([0, 2, 0, 1])
print(a[numpy.arange(4), b])

a[numpy.arange(4), b] += 10
print(a)

# Boolean array indexing
a = numpy.array([[1, 2], [3, 4], [5, 6]])
bool_idx = (a > 2)

print(bool_idx)

print(a[bool_idx])

x = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
y = numpy.array([[5, 6], [7, 8]], dtype=numpy.float64)

print(x + y)
print(numpy.add(x, y))

print(x - y)
print(numpy.subtract(x, y))

import numpy as np

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

v = np.array([9, 10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))

# Numpy provides many useful functions for performing computations on arrays; one of the most useful is sum:

x = np.array([[1, 2], [3, 4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"

x = np.array([[1, 2], [3, 4]])
print(x)
print(x.T)

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)  # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(y)

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()


# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()


# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()

