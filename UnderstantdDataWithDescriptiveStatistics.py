import pandas

url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)

# Understand your data using the head() function to look at the first few rows.
print(data.head())
# Review the dimensions of your data with the shape property.
print(data.shape)
# Look at the data types for each attribute with the dtypes property.
print(data.dtypes)
# Review the distribution of your data with the describe() function.
print(data.describe())
# Calculate pairwise correlation between your variables using the corr() function.
print(data.corr())
