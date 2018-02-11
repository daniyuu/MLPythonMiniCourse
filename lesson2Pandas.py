import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s = pd.Series([1, 3, 5, np.nan, 6, 8])

print(s)

# Create a DataFrame
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df)

df2 = pd.DataFrame({
    'A': 1.,
    'B': pd.Timestamp('20130101'),
    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
    'D': np.array([3] * 4, dtype='int32'),
    'E': pd.Categorical(['test', "train", "test", "train"]),
    'F': 'foo'
})
print(df2)

print(df.head())
print(df.head(2))
print(df.tail(2))
print(df.describe())
print(df.T)
print(df.sort_index(axis=1, ascending=False))
print(df.sort_values(by='B'))

# selection
print(df['A'])
print(df[0:3])
print(df['20130102':])

print(df.loc[dates[0], 'A'])

# Boolean Indexing
print(df[df.A > 0])

df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
print(df2[df2['E'].isin(['two', 'four'])])

# Setting
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))
df['F'] = s1
print(df)

# Operations
print(df.mean())
