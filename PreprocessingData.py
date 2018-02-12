import numpy as np
from sklearn import preprocessing

X_train = np.array([[1., -1., 2.],
                    [2., 0., 0.],
                    [0., 1., -1.]])

X_scaled = preprocessing.scale(X_train)
print(X_scaled)

subX = X_train[:,:2]
print(subX)
print(preprocessing.scale(subX))