import numpy as np
from sklearn import preprocessing

X_train = np.array([[1., -1., 2.],
                    [2., 0., 0.],
                    [0., 1., -1.]])

print(X_train)
X_scaled = preprocessing.scale(X_train)
print(X_scaled)

subX = X_train[:, :2]
print(subX)
print(preprocessing.scale(subX))

# Scaling features to a range
print("Scaling features to a range")
min_max_scaler = preprocessing.MinMaxScaler()
X_tran_minmax = min_max_scaler.fit_transform(X_train)
print(X_tran_minmax)

# MaxAbsScaler lies within the range [-1,1]
print("MaxAbsScaler lies within the range [-1,1]")
max_abs_scaler = preprocessing.MaxAbsScaler()
X_tran_maxabs = max_abs_scaler.fit_transform(X_train)
print(X_tran_maxabs)

# Non-linear transformation
print("Non-linear transformation")
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.fit_transform(X_test)
result = np.percentile(X_train[:, 0], [0, 25, 50, 75, 100])
print(result)
result_trans = np.percentile(X_train_trans[:, 0], [0, 25, 50, 75, 100])
print(result_trans)

quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
X_trans = quantile_transformer.fit_transform(X)
print(quantile_transformer.quantiles_)

# Normalization
print("Normalization")
X = [[1., -1., 2.],
     [2., 0., 0.],
     [0., 1., -1.]]
X_normalized = preprocessing.normalize(X, norm='l2')
print(X_normalized)

X = [[1., -1.],
     [2., 0.],
     [0., 1.]]
X_normalized = preprocessing.normalize(X, norm='l2')
print(X_normalized)

normalizer = preprocessing.Normalizer().fit(X)
print(normalizer.transform(X))

# Binarization
# Feature binarization
print("Feature binarization")
X = [[1., -1., 2.],
     [2., 0., 0.],
     [0., 1., -1.]]
binarizer = preprocessing.Binarizer().fit(X)
print(binarizer.transform(X))

# Imputation of missing values
print("Imputation of missing values")
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', verbose=0)
imp.fit([[1, 2], [np.nan, 3], [7, 6]])
X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))

# Generating polynomial features
print("Generating polynomial features")
from sklearn.preprocessing import PolynomialFeatures

X = np.arange(6).reshape(3, 2)
poly = PolynomialFeatures(2)
print(poly.fit_transform(X))

X = np.arange(9).reshape(3, 3)
poly = PolynomialFeatures(degree=3, interaction_only=True)
print(poly.fit_transform(X))
