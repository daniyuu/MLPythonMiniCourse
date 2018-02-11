# Scatter Plot Matrix
import matplotlib.pyplot as plt
import pandas
from pandas import scatter_matrix
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
plt.hist(data['preg'])
# scatter_matrix(data)
plt.show()
