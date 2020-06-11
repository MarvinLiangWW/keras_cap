
import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
iris = datasets.load_iris()
print(iris)
print(iris.data.shape, iris.target.shape)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.5, random_state=0)
print(X_train)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print(clf.score(X_test, y_test))




'''将所有的样本分成k组'''
'''
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import KFold
iris=datasets.load_iris()
kf = KFold(iris, n_folds=2)
for train, test in kf:
    print("%s %s" % (train, test))

'''
