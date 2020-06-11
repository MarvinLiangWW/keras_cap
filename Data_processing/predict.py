# import csv
# csv_reader = csv.reader(open('number-category.csv', encoding='utf-8'))
# f1=open('number.csv','w+')
# f2=open('category.csv','w+')
# for i in csv_reader:
#     f1.write(i[0]+'\n')
#     f2.write(i[1]+'\n')
# f1.close()
# f2.close()

# print(clf.score(X_test, y_test))
# scores = cross_val_score(clf,x,y,5)
# print(scores)
# print(mean(scores))

# print(classification_report(clf.predict(X_train), X_test, target_names=['1', '2', '3','4','5']))

import numpy as np
from numpy import mean
import csv
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.cross_validation import cross_val_score, KFold

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.externals import joblib

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import random

# joblib.dump(clf, 'filename.pkl')
# clf = joblib.load('filename.pkl')


def fun(filename1,filename2):
    csv_reader1 = csv.reader(open(filename1))
    csv1 = []
    for i in csv_reader1:
        temp = []
        a = 1
        while a < len(i):
            temp.append(float(i[a]))
            a += 1
        csv1.append(temp)
    # print(csv1)
    csv_reader2 = csv.reader(open(filename2))
    y = [int(i[0]) for i in csv_reader2]
    # print(y)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(csv1, y, test_size=0.2, random_state=0)
    return (X_train,X_test,y_train,y_test)


X_train2, X_test2, y_train2, y_test2 =fun('number_age_white_bxbzs_zxlxbbfb_lbxbbfb_cfydb_ALT_AST.csv','category.csv')
X_train, X_test, y_train, y_test =fun('number_age_white_bxbzs_zxlxbbfb_lbxbbfb_cfydb_ALT_AST_tiwen.csv','category.csv')

gbdtnew = GradientBoostingClassifier().fit(X_train,y_train)


def proba_fun(model,X_test,y_test):
    y_pred = model.predict(X_test)
    y_proba =model.predict_proba(X_test)
    counti =0
    for i in y_proba:
        countk = 0
        for k in i:
            if k > 0.6 and countk+1 ==y_test[counti]:
                print(i)
            countk+=1
        counti+=1
    # print(y_proba)



proba_fun(gbdtnew,X_test,y_test)

# clf = GaussianNB().fit(X_train, y_train)
# clf1 = tree.DecisionTreeClassifier().fit(X_train,y_train)
# clf9 = MultinomialNB(alpha=0.01).fit(X_train,y_train)
# clf2 = svm.SVC(gamma=0.001).fit(X_train,y_train)

#GBDT,svc,randomforseclassifier adjust the parameter add temp

# the result of the function LinerSVC() is changing
# clf4 = svm.LinearSVC().fit(X_train,y_train)

# clf3 = RandomForestClassifier().fit(X_train,y_train)
# clf13 = RandomForestClassifier(n_estimators=5,min_samples_split=3,min_samples_leaf=2).fit(X_train,y_train)
# clf23 = RandomForestClassifier(n_estimators=5,min_samples_split=3,min_samples_leaf=3).fit(X_train,y_train)
# clf32 = RandomForestClassifier().fit(X_train2,y_train2)
# clf213 = RandomForestClassifier(n_estimators=5,min_samples_split=3,min_samples_leaf=2).fit(X_train2,y_train2)
# clf223 = RandomForestClassifier(n_estimators=5,min_samples_split=3,min_samples_leaf=3).fit(X_train2,y_train2)

# micro : 0.279022403259
# macro : 0.249925956707
# micro : 0.295315682281
# macro : 0.267242809868
# micro : 0.313645621181
# macro : 0.284497348195
# micro : 0.262729124236
# macro : 0.235293089931
# micro : 0.26883910387
# macro : 0.228059920105
# micro : 0.281059063136
# macro : 0.238587378701

# GBDT = GradientBoostingClassifier(learning_rate=0.1,n_estimators=100,max_depth=3).fit(X_train,y_train) # 0.711666
# gbdt = GradientBoostingClassifier(learning_rate=0.01,n_estimators=1000).fit(X_train,y_train)  #1.0
# gbdt2 =GradientBoostingClassifier(learning_rate=0.01,n_estimators=100).fit(X_train,y_train) # 0.418237
# gbdt3 =GradientBoostingClassifier(learning_rate=0.1,max_depth=5).fit(X_train,y_train) # 0.989302
# gbdt4 =GradientBoostingClassifier(learning_rate=0.1,n_estimators=200).fit(X_train,y_train) # 0.876210
# gbdt5 =GradientBoostingClassifier(learning_rate=0.1,n_estimators=80).fit(X_train,y_train) # 0.663780
# gbdt6 =GradientBoostingClassifier(learning_rate=1.5,n_estimators=10,max_depth=3).fit(X_train,y_train) # 0.663780
#
# GBDTt = GradientBoostingClassifier(learning_rate=0.1,n_estimators=100,max_depth=3).fit(X_train2,y_train2) # 0.711666
# gbdtt = GradientBoostingClassifier(learning_rate=0.01,n_estimators=1000).fit(X_train2,y_train2)  #1.0
# gbdt2t =GradientBoostingClassifier(learning_rate=0.01,n_estimators=100).fit(X_train2,y_train2) # 0.418237
# gbdt3t =GradientBoostingClassifier(learning_rate=0.1,max_depth=5).fit(X_train2,y_train2) # 0.989302
# gbdt4t =GradientBoostingClassifier(learning_rate=0.1,n_estimators=200).fit(X_train2,y_train2) # 0.876210
# gbdt5t =GradientBoostingClassifier(learning_rate=0.1,n_estimators=80).fit(X_train2,y_train2) # 0.663780
# gbdt6t =GradientBoostingClassifier(learning_rate=1.5,n_estimators=10,max_depth=3).fit(X_train2,y_train2) # 0.663780

# nbc_1 = Pipeline([
#     ('vect', CountVectorizer()),
#     ('clf', MultinomialNB()),
# ])
# nbc_2 = Pipeline([
#     ('vect', HashingVectorizer(non_negative=True)),
#     ('clf', MultinomialNB()),
# ])
# nbc_3 = Pipeline([
#     ('vect', TfidfVectorizer()),
#     ('clf', MultinomialNB()),
# ])
#
# nbcs = [nbc_1, nbc_2, nbc_3]
#
# score=cross_validation.cross_val_score(nbc_1, X_train, y_train, cv= KFold(len(y_train), 5, shuffle=True, random_state=0))
# print(score)


# scores = cross_validation.cross_val_score(clf3, X_train, y_train, cv=5)
# print(scores)
#
# print(clf9.score(X_test,y_test))

#clf5 = linear_model.LogisticRegression(C=1e5)
#clf6 = ExtraTreesRegressor(n_estimators=10,random_state=0).fit(X_train,y_train)
#clf7 = KNeighborsRegressor().fit(X_train,y_train)
#clf8 = RidgeCV().fit(X_train,y_train)

def printresult(model,X_train,y_train,X_test,y_test):
    predict = [0, 0, 0, 0, 0]
    result = [0, 0, 0, 0, 0]
    true = [0, 0, 0, 0, 0]
    a = 0
    while a < len(X_test):
        temp = model.predict(np.array(X_test[a]).reshape(1,-1)) - 1
        predict[temp] += 1
        result[y_test[a] - 1] += 1
        if temp + 1 == y_test[a]:
            true[temp] += 1
        a += 1
    # print(predict)
    # print(result)
    # print(true)
    # print(true[0] / result[0])
    # print(true[1] / result[1])
    # print(true[2] / result[2])
    # print("train score :%f"%(model.score(X_train,y_train)))
    # print(model.score(X_test,y_test))
    y_pred =model.predict(X_test)
    # print(y_pred)
    print('micro :',f1_score(y_test,y_pred,average='micro'))
    print('macro :',f1_score(y_test,y_pred,average='macro'))


# printresult(clf3,X_train,y_train,X_test,y_test)
# printresult(clf32,X_train2, y_train2,X_test2,  y_test2 )
# printresult(clf13,X_train,y_train,X_test,y_test)
# printresult(clf213,X_train2,y_train2,X_test2,y_test2)
# printresult(clf23,X_train,y_train,X_test,y_test)
# printresult(clf223,X_train2,y_train2,X_test2,y_test2)

# printresult(GBDT,X_train,y_train,X_test,y_test)
# printresult(gbdt,X_train,y_train,X_test,y_test)
# printresult(gbdt2,X_train,y_train,X_test,y_test)
# printresult(gbdt3,X_train,y_train,X_test,y_test)
# printresult(gbdt4,X_train,y_train,X_test,y_test)


# n_estimator_params = range(1, 100,1)
# count =0
# k=0
# for n_estimator in n_estimator_params:
#     rf = RandomForestClassifier(n_estimators=n_estimator,n_jobs=-1, verbose=True)
#     rf.fit(X_train, y_train)
#     a=(rf.predict(X_test) == y_test).mean()
#     if a>count:
#         count =a
#         k=n_estimator
#     print ("Accuracy:\t", a)
#
# print(count,k)

# print('estimators:',clf3.estimators_)
# print('classes_:',clf3.classes_)
# print('n_classes_:',clf3.n_classes_)
# print('n_features;',clf3.n_features_)
# print('n_outputs_',clf3.n_outputs_)
# print('oob_score_',clf3.oob_score_)
# print('oob_decision_function_',clf3.oob_decision_funcition_)


def printF1score(model,X_test,y_test):
    y_pred = model.predict_proba(X_test)
    # print(y_pred)
    # print('micro :', f1_score(y_test, y_pred, average='micro'))
    # print('macro :', f1_score(y_test, y_pred, average='macro'))
    print('accuracy:',accuracy_score(y_test,y_pred))

# printF1score(clf3,X_test,y_test)
# printF1score(clf32,X_test2,  y_test2 )
# printF1score(clf13,X_test,y_test)
# printF1score(clf213,X_test2,y_test2)
# printF1score(clf23,X_test,y_test)
# printF1score(clf223,X_test2,y_test2)

# printF1score(GBDT,X_test,y_test) #micro : 0.309572301426  macro : 0.268436991114 0.31
# printF1score(gbdt,X_test,y_test) #micro : 0.321792260692  macro : 0.273254625276  0.32
# printF1score(gbdt2,X_test,y_test) # micro : 0.354378818737  macro : 0.275858605433 0.35
# printF1score(gbdt3,X_test,y_test)
# printF1score(gbdt4,X_test,y_test)
# printF1score(gbdt5,X_test,y_test) #micro : 0.301425661914 macro : 0.250546575407
# printF1score(gbdt6,X_test,y_test)
#
# printF1score(GBDTt,X_test2,y_test2) #micro : 0.325865580448 macro : 0.274369520837 0.30
# printF1score(gbdtt,X_test2,y_test2)# 0.33
# printF1score(gbdt2t,X_test2,y_test2)
# printF1score(gbdt3t,X_test2,y_test2) #micro : 0.334012219959 macro : 0.296546485469
# printF1score(gbdt4t,X_test2,y_test2)
# printF1score(gbdt5t,X_test2,y_test2)
# printF1score(gbdt6t,X_test2,y_test2)
#
#


def randomoutput(X_test,y_test):
    y_pred=[]
    for i in X_test:
        y_pred.append(random.randint(1, 5))
    print('micro :', f1_score(y_test, y_pred, average='micro'))
    print('macro :', f1_score(y_test, y_pred, average='macro'))
    print('accuracy:', accuracy_score(y_test, y_pred))

# randomoutput(X_test,y_test)


def mostlabel(X_test,y_test):
    y_pred=[]
    for i in X_test:
        y_pred.append(5)
    print('micro :', f1_score(y_test, y_pred, average='micro'))
    print('macro :', f1_score(y_test, y_pred, average='macro'))
    print('accuracy:', accuracy_score(y_test, y_pred))
    y1 =[]
    y2 =[]
    y3 =[]
    y4 =[]
    y5 = []
    for i in X_test:
        y1.append(1)
        y2.append(2)
        y3.append(3)
        y4.append(4)
        y5.append(5)
    print('accuracy:', accuracy_score(y_test, y1))
    print('accuracy:', accuracy_score(y_test, y2))
    print('accuracy:', accuracy_score(y_test, y3))
    print('accuracy:', accuracy_score(y_test, y4))
    print('accuracy:', accuracy_score(y_test, y5))

# mostlabel(X_test,y_test)