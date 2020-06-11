import csv
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
import random
import matplotlib.pyplot as plt


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

    csv_reader2 = csv.reader(open(filename2))
    y = [int(i[0]) for i in csv_reader2]
    return csv1,y


def printscore(model,X_test,y_test):
    y_pred = model.predict(X_test)
    # print('micro :', f1_score(y_test, y_pred, average='micro'))
    # print('macro :', f1_score(y_test, y_pred, average='macro'))
    # print('accuracy:',accuracy_score(y_test,y_pred))
    return accuracy_score(y_test,y_pred)

    # return accuracy_score(y_test,y_pred)


def testSelectKBest():
    X, y = fun('final.csv', 'Category.csv')
    ac1 = 0
    ac2 = 0
    ac3 = 0
    i1 = 0
    i2 = 0
    i3 = 0
    for i in range(2, 31):
        X_new = SelectKBest(chi2, i).fit_transform(X, y)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_new, y, test_size=0.2, random_state=0)
        rfc = RandomForestClassifier().fit(X_train, y_train)
        svc = LinearSVC().fit(X_train, y_train)
        gbdt = GradientBoostingClassifier(n_estimators=100).fit(X_train, y_train)
        if printscore(rfc, X_test, y_test) > ac1:
            i1 = i
            ac1 = printscore(rfc, X_test, y_test)
        if printscore(svc, X_test, y_test) > ac2:
            i2 = i
            ac2 = printscore(svc, X_test, y_test)
        if printscore(gbdt, X_test, y_test) > ac3:
            i3 = i
            ac3 = printscore(gbdt, X_test, y_test)
    print("%f %f\n %f %f\n %f %f\n" % (i1, ac1, i2, ac2, i3, ac3))

#  testSelectKBest result
#  29.000000 0.323741  33
#  20.000000 0.291367   32
#  13.000000 0.356115   35

if __name__=="__main__":
    testSelectKBest()

    X, y = fun('final.csv', 'Category.csv')

    svc = LinearSVC(penalty='l1',dual=False).fit(X, y)
    model = SelectFromModel(svc,prefit=True)
    X_new =model.transform(X)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_new, y, test_size=0.2, random_state=0)
    rfc = RandomForestClassifier().fit(X_train, y_train)
    gbdt = GradientBoostingClassifier(n_estimators=120).fit(X_train, y_train)
    print(printscore(rfc,X_train,y_train))
    print(printscore(gbdt,X_train,y_train))



    # clf = Pipeline([
    #     ('feature_selection',SelectFromModel(LinearSVC())),
    #     ('classification',RandomForestClassifier())
    # ])
    # clf.fit(X_train,y_train)
    # printscore(clf,X_test,y_test)
    #
    # rf = RandomForestClassifier().fit(X_train,y_train)
    # # printscore(rf,X_test,y_test)
    # importance1 = rf.feature_importances_
    # indict1 = np.argsort(importance1)[::-1]
    # for i in range(0, 30):
    #     print("%d(%f)" % (indict1[i], importance1[indict1[i]]))
    #
    # et = ExtraTreesClassifier(n_estimators=100).fit(X_train,y_train)
    # printscore(et,X_test,y_test)
    # importance = et.feature_importances_
    # indict = np.argsort(importance)[::-1]
    # for i in range (0,30):
    #     print("%d(%f)"%(indict[i],importance[indict[i]]))

    # svc = LinearSVC()
    # rfecv = RFECV(estimator=svc,scoring='accuracy').fit(X,y)
    # print(rfecv.n_features_)
    # plt.figure()
    # plt.plot(range(1,len(rfecv.grid_scores_)+1),rfecv.grid_scores_)
    # plt.show()

    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_new, y, test_size=0.2)
    # gbdt = GradientBoostingClassifier().fit(X_train, y_train)
    # printscore(gbdt, X_test, y_test)
    # plt.figure()
    # score_train = list()
    # score_test =list()
    # k =[]
    # for i in range(1,11):
    #     k.append(i)
    #     X_train, X_test, y_train, y_test = fun('number_age_col71.csv', 'Category.csv')
    #     gbdt = GradientBoostingClassifier().fit(X_train, y_train)
    #     score_test.append(accuracy_score(y_test,gbdt.predict(X_test)))
    #     score_train.append(accuracy_score(y_train,gbdt.predict(X_train)))
    # plt.errorbar(k, score_test)
    # plt.errorbar(k,score_train)
    # plt.axis('tight')
    # plt.show()

    # rfc =RandomForestClassifier().fit(X_train,y_train)
    # printscore(rfc,X_test,y_test)

    # X_new =GenericUnivariateSelect(mode='k').fit_transform(csv1, y)
    # X_new = SelectKBest(chi2, k=10).fit_transform(csv1, y)
    # print(X_new)
    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_new, y, test_size=0.2)


