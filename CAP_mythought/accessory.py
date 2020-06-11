import csv
from datetime import datetime,timedelta
import numpy as np
import random
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from keras.layers import Dropout,Dense,Activation
from matplotlib.pyplot import savefig,plot,legend,show,title,subplot,figure,ylim
from keras.optimizers import Adam,SGD
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
random.seed(1337)
nb_classes =2
data_version =5
nb_epochs=200
batch_size =16

def diff_length_dat(filename):
    f_in = open(filename)
    return_list=[]
    for i,lines in enumerate(f_in):
        line = lines.strip().split(' ')[0:]
        return_list.append(line)
    return return_list

def diff_length_csv(filename):
    f_in = open(filename)
    return_list = []
    for i, lines in enumerate(f_in):
        line = lines.strip().split(',')[0:]
        return_list.append(line)
    return return_list

def same_length_csv(filename):
    csv_reader=csv.reader(open(filename))
    csv1=[]
    for i in csv_reader:
        temp=[]
        a=0
        while a<len(i):
            temp.append(float(i[a]))
            a+=1
        csv1.append(temp)
    return np.array(csv1,dtype=type(csv1[0][0]))

def read_case_nb(f_in='nb_x_train.dat'):
    f_in =open(f_in)
    rt_list=[]
    for i,lines in enumerate(f_in):
        line =lines.strip()
        rt_list.append(line)
    return rt_list

def category_to_target(category):
    y =[]
    for i in range(0,category.shape[0]):
        temp=[]
        for k in range(0,nb_classes):
            if k+1==category[i]:
                temp.append(1)
            else:
                temp.append(0)
        y.append(temp)
    return np.array(y,dtype=type(y[0][0]))

def get_train_test_data(X_con,y,nb_x_train,nb_x_test,):
    X_train, X_test, y_train, y_test, = [], [], [], [],
    for m in range(0, len(nb_x_train)):
        for n in range(0, len(X_con)):
            if float(nb_x_train[m]) == float(X_con[n][0]):
                X_train.append(X_con[n][1:])
                break
    for m in range(0, len(nb_x_train)):
        for n in range(0, len(y)):
            if float(nb_x_train[m]) == float(y[n][0]):
                y_train.append(y[n][1:])
                break
    for m in range(0, len(nb_x_test)):
        for n in range(0, len(X_con)):
            if float(nb_x_test[m]) == float(X_con[n][0]):
                X_test.append(X_con[n][1:])
                break
    for m in range(0, len(nb_x_test)):
        for n in range(0, len(y)):
            if float(nb_x_test[m]) == float(y[n][0]):
                y_test.append(y[n][1:])
                break
    return np.array(X_train,), np.array(X_test), np.array(y_train), np.array(y_test),

def logistic_regression_prediction(data_version,penalty='l1'):
    X = diff_length_csv('temperature.csv')
    X = pad_sequences(X, maxlen=50, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    X2 = same_length_csv('cap_feature.csv')
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_cv%d.dat'%(data_version))
    y = same_length_csv('number_category.csv')
    X_con = np.concatenate((X2, X), axis=1)
    X_train, X_test, y_train, y_test, = get_train_test_data(X_con, y, nb_x_train, nb_x_test, )
    lr =LogisticRegression(penalty=penalty,fit_intercept=True,max_iter=200,tol=0.00005)
    lr =lr.fit(X_train,y_train,)
    # print(lr.coef_)
    score =lr.score(X_test,y_test)
    print(score)
    return score

def logistic_regression_prediction_epoch(data_version,penalty='l1'):
    X = diff_length_csv('temperature.csv')
    X = pad_sequences(X, maxlen=50, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    X2 = same_length_csv('cap_feature.csv')
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_cv%d.dat'%(data_version))
    y = same_length_csv('number_category.csv')
    X_con = np.concatenate((X2, X), axis=1)
    X_train, X_test, y_train, y_test, = get_train_test_data(X_con, y, nb_x_train, nb_x_test, )
    acc_list = []
    for i in range(nb_epochs):
        lr = LogisticRegression(penalty=penalty, fit_intercept=True, max_iter=i, warm_start=True)
        lr = lr.fit(X_train, y_train, )
        score = lr.score(X_test, y_test)
        acc_list.append(score)
    plot(range(0, nb_epochs), acc_list, label='temp_acc')
    acc_list_100_110 = acc_list[99:109]
    acc_list_200 = acc_list[0:200]
    print(len(acc_list_200))
    acc_list_210 = acc_list[200:]
    print(acc_list_210)
    print(len(acc_list_210))
    acc_list_sored = sorted(acc_list_200, reverse=True)
    print(acc_list)
    title('temp_study_%d' % (data_version))
    print('temp_study_%d\n' % (data_version))
    print("top-10 mean: %.3f" % np.mean(np.array(acc_list_sored[:10])))
    print("top-50 mean: %.3f" % np.mean(np.array(acc_list_sored[:50])))
    # print("last-10 mean: %.3f" % np.mean(np.array(acc_list_210)))
    print("acc_100-110 mean: %.3f" % np.mean(np.array(acc_list_100_110)))

def logistic_regression_feature_prediction(data_version,penalty ='l1'):
    # X2 = fun2('number_age_col85tran_v2.csv')
    # X2 = fun2('number_age_col71tran_v2.csv')
    X2 = same_length_csv('cap_feature.csv')
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_cv%d.dat'%(data_version))
    y = same_length_csv('number_category.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X2, y, nb_x_train, nb_x_test, )
    lr = LogisticRegression(penalty=penalty, fit_intercept=True, max_iter=200, warm_start=True,tol=0.0001)
    lr = lr.fit(X_train, y_train, )
    # print(lr.coef_)
    score = lr.score(X_test, y_test)
    print(score)
    return score

def logistic_regression_temperature_prediction(data_version,penalty ='l1'):
    X = diff_length_csv('temperature.csv')
    X = pad_sequences(X, maxlen=50, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_cv%d.dat'%(data_version))
    y = same_length_csv('number_category.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    lr = LogisticRegression(penalty=penalty, fit_intercept=True, max_iter=200, warm_start=True)
    lr = lr.fit(X_train, y_train, )
    # print(lr.coef_)
    score = lr.score(X_test, y_test)
    print(score)
    return score

def logistic_regression_temprature_prediction_epoch(data_version,penalty ='l1'):
    X = diff_length_csv('temperature.csv')
    X = pad_sequences(X, maxlen=50, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_cv%d.dat'%(data_version))
    y = same_length_csv('number_category.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    acc_list=[]
    for i in range(nb_epochs):
        lr = LogisticRegression(penalty=penalty, fit_intercept=True, max_iter=i, warm_start=True)
        lr = lr.fit(X_train, y_train, )
        score = lr.score(X_test, y_test)
        acc_list.append(score)
    plot(range(0, nb_epochs), acc_list, label='temp_acc')
    acc_list_100_110 = acc_list[99:109]
    acc_list_200 = acc_list[0:200]
    print(len(acc_list_200))
    acc_list_210 = acc_list[200:]
    print(acc_list_210)
    print(len(acc_list_210))
    acc_list_sored = sorted(acc_list_200, reverse=True)
    print(acc_list)
    title('temp_study_%d' % (data_version))
    print('temp_study_%d\n' % (data_version))
    print("top-10 mean: %.3f" % np.mean(np.array(acc_list_sored[:10])))
    print("top-50 mean: %.3f" % np.mean(np.array(acc_list_sored[:50])))
    # print("last-10 mean: %.3f" % np.mean(np.array(acc_list_210)))
    print("acc_100-110 mean: %.3f" % np.mean(np.array(acc_list_100_110)))

def gbdt_temprature_prediction(data_version):
    X = diff_length_csv('temperature.csv')
    X = pad_sequences(X, maxlen=50, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_cv%d.dat'%(data_version))
    y = same_length_csv('number_category.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )

    lr = GradientBoostingClassifier(n_estimators=100,learning_rate=0.01,max_depth=3)
    lr = lr.fit(X_train, y_train, )
    score = lr.score(X_test, y_test)
    print(score)
    return score


def print_score():
    score = 0.0
    for data_version in range(1, 6):
        # score += logistic_regression_temperature_prediction(data_version=data_version, penalty='l1')
        # score += logistic_regression_temperature_prediction(data_version=data_version,penalty='l2')
        # score += logistic_regression_feature_prediction(data_version=data_version,penalty='l1')
        # score += logistic_regression_feature_prediction(data_version=data_version,penalty='l2')
        # score += logistic_regression_prediction(data_version=data_version,penalty='l1')
        # score += logistic_regression_prediction(data_version=data_version,penalty='l2')
        score += gbdt_temprature_prediction(data_version=data_version,)
    print(score /5)

def nb_5_cv_split(f_in ='number.csv',nb_cv =1):
    f_in =open(f_in)
    f_train_cv1 =open('nb_train_cv%d.dat'%(nb_cv),'w')
    f_test_cv1 =open('nb_test_cv%d.dat'%(nb_cv),'w')
    for i,lines in enumerate(f_in):
        line =lines.strip()
        t =i%5
        if t !=nb_cv-1:
            f_train_cv1.write(line+'\n')
        else:
            f_test_cv1.write(line+'\n')


def category_split(f_in ='number_category.csv',category =1):
    f_in =open(f_in)
    f_out =open('number_category_%d.dat'%(category),'w')
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')
        if line[1]==str(category):
            f_out.write(line[0]+'\n')






if __name__ =='__main__':
    # category_split(category=1)
    # category_split(category=2)
    # nb_5_cv_split(nb_cv=1)
    # nb_5_cv_split(nb_cv=2)
    # nb_5_cv_split(nb_cv=3)
    # nb_5_cv_split(nb_cv=4)
    # nb_5_cv_split(nb_cv=5)
    print_score()

    # for data_version in range(2,3):
        # logistic_regression_prediction_epoch(data_version=data_version,penalty='l2')
        # logistic_regression_temprature_prediction_epoch(data_version=data_version,penalty='l2')
        # ylim((0.5,1.0))
        # show()





