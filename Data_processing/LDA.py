import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf=LinearDiscriminantAnalysis( solver='svd')
clf.fit(X, y)
print(clf.predict([[1,3]]))
print(X.shape)


k = [[[1,1],],[[2,2],2],[[3,3]],3]
a = np.array([[k[0],1],[k[1],2],[k[2],3]])
b = np.array([1,2,3])
ab = GaussianNB().fit(k,b)
ab.predict([[[4,4],4]])


cd = tree.DecisionTreeClassifier() .fit(X,y)
print(cd.predict([[0,0]]))