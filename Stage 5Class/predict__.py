import csv
from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import chi2
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import numpy as np
import random
np.random.seed(40)

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

def fun2(filename):
    csv_reader1 = csv.reader(open(filename))
    csv1 = []
    for i in csv_reader1:
        temp = []
        a = 1
        while a < len(i):
            temp.append(float(i[a]))
            a += 1
        csv1.append(temp)
    return csv1

def gbdt_predict(X,y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    gbdt = GradientBoostingClassifier().fit(X_train, y_train)
    return metrics.accuracy_score(y_test, gbdt.predict(X_test))

def rfc_predict(X,y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    rfc = RandomForestClassifier().fit(X_train, y_train)
    return metrics.accuracy_score(y_test, rfc.predict(X_test))

def cross_val_prediction(X,y):
    gbdt =GradientBoostingClassifier()
    scores =cross_val_score(gbdt,X,y,cv =5)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print(scores)
    return scores.mean()

def most_label(y):
    count=[0,0,0]
    for i in range(0,len(y)):
        count[int(y[i])-1]+=1
    print(count[0]/len(y))
    print(count[1]/len(y))
    print(count[2]/len(y))

def random_guess(y):

    acc =0.0
    for k in range(0,10):
        y_pred = []
        for i in range(0,len(y)):
            y_pred.append(random.randint(1,3))
        acc+=accuracy_score(y,y_pred)
    print(acc)

def chi2_best_feature(X,y,mode):
    best = []
    count = []
    output = 'chi2_best_feature'
    if mode ==0:
        output+='.dat'
    elif mode ==1:
        output+='tiwen.dat'
    f = open(output,'w')
    for i in range(3,len(X[0])):
        k = 0.0
        X_new = SelectKBest(chi2, i).fit_transform(X,y)
        for t in range(0, 10):
            k+=cross_val_prediction(X_new,y)
        if k>5.4:
            best.append(k)
            count.append(i)
        f.write('%d %.7f\n'%(i,k))
    print(best)
    print(count)
    f.close()

def chi2_feature(X,y,count):
    for i in range(0,len(count)):
        acc=0
        X_new = SelectKBest(chi2, count[i]).fit_transform(X, y)
        for k in range(0,10):
            acc+=gbdt_predict(X_new, y)
        print(acc)

def etc_best_feature(X,y,mode):
    best = []
    count = []
    output = 'etc_best_feature_stdtran'
    if mode == 0:
        output += '.dat'
    elif mode == 1:
        output += 'tiwen.dat'
    f = open(output, 'w')
    for i in range(3, len(X[0])):
        k = 0.0
        etc =ExtraTreesClassifier().fit(X,y)
        X_new = SelectFromModel(etc,prefit=True).transform(X)
        for t in range(0, 10):
            k += cross_val_prediction(X_new, y)
        if k > 5.4:
            best.append(k)
            count.append(i)
        f.write('%d %.7f\n' % (i, k))
    print(best)
    print(count)
    f.close()

def etc_gbdt(X,y,mode):
    f=open('etc_feature.csv',mode=mode)
    k = 0.0
    etc = ExtraTreesClassifier().fit(X, y)
    X_new = SelectFromModel(etc, prefit=True).transform(X)
    print(X_new[0])
    f.write(str(X_new[0]))
    f.write('\n')
    gbdt = GradientBoostingClassifier().fit(X_new, y)
    for t in range(0, 10):
        k += cross_val_prediction(X_new, y)
    print(k.mean())
    importance1 = gbdt.feature_importances_
    indict1 = np.argsort(importance1)[::-1]
    for i in range(0, len(indict1)):
        print("%d(%f)" % (indict1[i], importance1[indict1[i]]))


def length_of_file(file):
    csv_reader = csv.reader(open(file))
    for i in csv_reader:
        print(len(i))
        break


def feature_importance(X,y,mode):
    k = 0.0
    X_new = SelectKBest(chi2, 25).fit_transform(X, y)
    print(X_new[0])
    print(X_new[1])
    gbdt = GradientBoostingClassifier().fit(X_new,y)
    for t in range(0, 10):
        k += cross_val_score(gbdt, X_new, y, cv=5)
    print(k)
    print(k.mean())
    importance1 = gbdt.feature_importances_
    indict1 = np.argsort(importance1)[::-1]
    for i in range(0, 25):
        print("%d(%f)" % (indict1[i], importance1[indict1[i]]))

def tiwen_predict(f_in='number_age_col71tran.csv'):
    X, y = fun(f_in, 'Category.csv')
    gbdt = GradientBoostingClassifier().fit(X,y)
    k=0
    for t in range(0, 10):
        k += cross_val_score(gbdt, X, y, cv=5)
    print(k)
    print(k.mean())
    importance1 = gbdt.feature_importances_
    indict1 = np.argsort(importance1)[::-1]
    for i in range(0, len(indict1)):
        print("%d(%f)" % (indict1[i], importance1[indict1[i]]))




if __name__ == '__main__':
    print("main")
    tiwen_predict('tiwencheck.csv')
    tiwen_predict('5_day_50_check.csv')

    X, y =fun('number_age_col71tran.csv', 'Category.csv')
    X2 = fun2('tiwenlabel.csv')
    # print(gbdt_predict(X,y))
    # print(rfc_predict(X,y))
    # most_label(y)
    # random_guess(y)
    # chi2_best_feature(X,y,0)
    # etc_best_feature(X,y,0)
    # length_of_file('number_age_col71.csv')

    etc_gbdt(X,y,'w')


    X_new = np.concatenate((X, X2), axis=1)
    print(X_new[0])
    # print(gbdt_predict(X_new,y))
    # print(rfc_predict(X_new,y))
    # chi2_best_feature(X_new,y,1)
    # etc_best_feature(X_new,y,1)
    # k=0.0
    # j=0.0
    # for i in range(0,10):
    #     k+=cross_val_prediction(X, y)
    #     j+=cross_val_prediction(X_new,y)
    # print(k)
    # print(j)

    etc_gbdt(X_new,y,'a')





