# Cross Validation Classification LogLoss
#
# url = "https://goo.gl/vhm1eU"
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# dataframe = read_csv(url, names=names)
# array = dataframe.values
# X = array[:, 0:8]
# Y = array[:, 8]
# kfold = KFold(n_splits=10, random_state=7)
# model = LogisticRegression()
# scoring = 'neg_log_loss'
# results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# print(("Logloss: %.3f (%.3f)") % (results.mean(), results.std()))

# Model Evaluation
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
X, y = iris.data, iris.target
clf = svm.SVC(probability=True, random_state=0)
results = cross_val_score(clf, X, y, scoring='neg_log_loss')
print(results)

# Accuracy score
from sklearn.metrics import accuracy_score

y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
res = accuracy_score(y_true, y_pred)
print(res)

# Cohen's kappa
from sklearn.metrics import cohen_kappa_score

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
res = cohen_kappa_score(y_true, y_pred)
print(res)

# Confusion matrix
from sklearn.metrics import confusion_matrix

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
res = confusion_matrix(y_true, y_pred)
print(res)

# For binary problems
y_true = [0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 1, 0, 1, 0, 1]
res = confusion_matrix(y_true, y_pred)
print(res)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(tn, fp, fn, tp)

# Classification report
from sklearn.metrics import classification_report

y_true = [0, 1, 2, 2, 0]
y_pred = [0, 0, 2, 1, 0]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))

# Hamming loss
from sklearn.metrics import hamming_loss

y_pred = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]
res = hamming_loss(y_true, y_pred)
print(res)

# Jaccard similarity coefficient score
from sklearn.metrics import jaccard_similarity_score

y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
res = jaccard_similarity_score(y_true, y_pred)

# Hinge loss

from sklearn import svm
from sklearn.metrics import hinge_loss

X = [[0], [1]]
y = [-1, 1]
est = svm.LinearSVC(random_state=0)
est.fit(X, y)
pred_decision = est.decision_function([[-2], [3], [0.5]])
print(pred_decision)
res = hinge_loss([-1, 1, 1], pred_decision)
print(res)

# Zero one loss
from sklearn.metrics import zero_one_loss
import numpy as np

y_pred = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]
res = zero_one_loss(y_true, y_pred)
print(res)
# In the multilabel case with binary label indicators
res = zero_one_loss(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
print(res)

# Log loss
from sklearn.metrics import log_loss

y_true = [0, 0, 1, 1]
y_pred = [[.9, .1], [.8, .2], [.3, .7], [.01, .99]]
res = log_loss(y_true, y_pred)
print(res)

# Matthews correlation coefficient
from sklearn.metrics import matthews_corrcoef

y_true = [+1, +1, +1, -1]
y_pred = [+1, -1, +1, +1]
res = matthews_corrcoef(y_true, y_pred)
print(res)

# Receiver operating characteristic (ROC)
from sklearn.metrics import roc_curve
