from sklearn import linear_model
X= [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]
clf = linear_model.LinearRegression()
clf.fit(X, y)
print(clf.coef_)
print(clf.intercept_)
print(clf.predict([[3, 3]]))
print(clf.predict([[5, 7.2]]))
