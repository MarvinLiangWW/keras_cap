from sklearn import tree
X = [[0, 0,0], [2, 2,2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf.fit(X, y)
print(clf.predict([[4, 4,4]]))
a=[[0,0],[1,1]]
b=[0,1]
ccc=tree.DecisionTreeClassifier()
ccc.fit(a,b)
print(ccc.predict([[4,4]]))