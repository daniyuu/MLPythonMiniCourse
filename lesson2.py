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
