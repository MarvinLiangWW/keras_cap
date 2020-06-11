from sklearn import svm
from sklearn import datasets
clf = svm.SVC(gamma=0.001)
iris = datasets.load_iris()
digits = datasets.load_digits()
X, y = iris.data, iris.target
clf.fit(digits.data[:-1], digits.target[:-1])
print(clf.predict(digits.data[0]))


