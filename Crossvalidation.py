# a random split into training and test sets
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

for c in range(1, 3):
    clf = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
    print(clf.score(X_test, y_test))

# Use other cross validation strategies by passing a cross validation iterator instead
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

n_samples = iris.data.shape[0]
cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
results = cross_val_score(clf, iris.data, iris.target, cv=cv)
print(results)

# Pipeline makes it easier to compose estimators, providing this behavior under cross-validation
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing

clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
results = cross_val_score(clf, iris.data, iris.target, cv=cv)
print(results)
