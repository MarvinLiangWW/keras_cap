import csv
from datetime import datetime,timedelta
import numpy as np
import random
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from keras.layers import Dropout,Dense,Activation
from matplotlib.pyplot import savefig,plot,legend,show,title,subplot,figure,ylim
import matplotlib.pyplot as plt
from keras.optimizers import Adam,SGD
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier,RandomForestClassifier
from keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
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

def diff_length_csv_from1(filename):
    f_in = open(filename)
    return_list = []
    for i, lines in enumerate(f_in):
        line = lines.strip().split(',')[1:]
        return_list.append(line)
    return return_list

def diff_length_csv_from2(filename):
    f_in = open(filename)
    return_list = []
    for i, lines in enumerate(f_in):
        line = lines.strip().split(',')[1:]
        line =np.asarray(line,dtype=float)
        return_list.append(list(line))
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
    X = diff_length_csv_from1('temperature_37.2_40.0.csv')
    X = pad_sequences(X, maxlen=time_step, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    # print(X.shape)
    X2 = same_length_csv('cap_feature.csv')
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_cv%d.dat'%(data_version))
    y = same_length_csv('number_category.csv')
    X_con = np.concatenate((X2, X), axis=1)
    # print(X_con[0])
    # print(nb_x_train[0])
    # print(X_con[0].shape)
    X_train, X_test, y_train, y_test, = get_train_test_data(X_con, y, nb_x_train, nb_x_test, )
    # print(X_train[0])
    lr = LogisticRegression(penalty=penalty, fit_intercept=True, max_iter=200, warm_start=True, tol=0.00005)
    lr = lr.fit(X_train, y_train, )
    # predicted =lr.predict(X_test)
    # f_out =open('lr_l2_predicted.csv','a')
    # for i in range(0,len(nb_x_test)):
    #     f_out.write('%s,%d\n'%(nb_x_test[i],predicted[i]))
    score = lr.score(X_test, y_test)
    y_pre = lr.predict(X_test)
    roc_auc = metrics.roc_auc_score(y_test-1, y_pre-1)
    print(score)
    return score,roc_auc

def logistic_regression_prediction_epoch(data_version,penalty='l1'):
    X = diff_length_csv('temperature_37.2_40.0.csv')
    X = pad_sequences(X, maxlen=time_step-1, padding='post', truncating='post', value=0, dtype=float)
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
    X2 = same_length_csv('cap_feature.csv')
    nb_x_train =read_case_nb(f_in ='nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_cv%d.dat'%(data_version))
    y = same_length_csv('number_category.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X2, y, nb_x_train, nb_x_test, )
    lr = LogisticRegression(penalty=penalty, fit_intercept=True, max_iter=200, warm_start=True,tol=0.0001)
    lr = lr.fit(X_train, y_train, )
    print(lr.coef_)
    score = lr.score(X_test, y_test)
    y_pre = lr.predict(X_test)
    roc_auc = metrics.roc_auc_score(y_test-1, y_pre-1)
    print(score)
    return score,roc_auc

def logistic_regression_temperature_prediction(data_version,penalty ='l1'):
    X = diff_length_csv('temperature_37.2_40.0.csv')
    X = pad_sequences(X, maxlen=time_step+1, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_cv%d.dat'%(data_version))
    y = same_length_csv('number_category.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    print(X_train.shape)
    lr = LogisticRegression(penalty=penalty, fit_intercept=True, max_iter=200,tol=0.00001)
    lr = lr.fit(X_train, y_train, )
    # print(lr.coef_)
    score = lr.score(X_test, y_test)
    y_pre = lr.predict(X_test)
    roc_auc = metrics.roc_auc_score(y_test-1, y_pre-1)
    print(score)
    return score,roc_auc

def logistic_regression_temprature_prediction_epoch(data_version,penalty ='l1'):
    X = diff_length_csv('temperature_37.2_40.0.csv')
    X = pad_sequences(X, maxlen=time_step-1, padding='post', truncating='post', value=0, dtype=float)
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
    X = diff_length_csv('temperature_37.2_40.0.csv')
    X = pad_sequences(X, maxlen=time_step+1, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_cv%d.dat'%(data_version))
    y = same_length_csv('number_category.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )

    # y_train = category_to_target(y_train)
    # y_test = category_to_target(y_test)
    lr = GradientBoostingClassifier(n_estimators=100,learning_rate=0.01,max_depth=3)
    lr = lr.fit(X_train, y_train, )
    score = lr.score(X_test, y_test)
    # print(lr.feature_importances_)
    y_pre = lr.predict(X_test)
    roc_auc = metrics.roc_auc_score(y_test-1, y_pre-1)
    print(score)
    return score,roc_auc

def rf_temprature_prediction(data_version):
    X = diff_length_csv('temperature_37.2_40.0.csv')
    X = pad_sequences(X, maxlen=time_step+1, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_cv%d.dat'%(data_version))
    y = same_length_csv('number_category.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    # print(y_train.shape)
    lr = RandomForestClassifier(n_estimators=100,max_depth=3)
    lr = lr.fit(X_train, y_train, )
    score = lr.score(X_test, y_test)
    y_pre = lr.predict(X_test)
    roc_auc = metrics.roc_auc_score(y_test-1, y_pre-1)
    print(score)
    return score,roc_auc

def gbdt_feature_prediction(data_version):
    X2 = same_length_csv('cap_feature.csv')
    nb_x_train = read_case_nb(f_in='nb_train_cv%d.dat' % (data_version))
    nb_x_test = read_case_nb(f_in='nb_test_cv%d.dat' % (data_version))
    y = same_length_csv('number_category.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X2, y, nb_x_train, nb_x_test, )
    lr = GradientBoostingClassifier(n_estimators=100,learning_rate=0.01,max_depth=3)
    lr = lr.fit(X_train, y_train, )
    # predicted =lr.predict(X_test)
    # f_out =open('gbdt_feature_predicted.csv','a')
    # for i in range(0,len(nb_x_test)):
    #     f_out.write('%s,%d\n'%(nb_x_test[i],predicted[i]))

    score = lr.score(X_test, y_test)
    print(lr.feature_importances_)
    y_pre = lr.predict(X_test)
    roc_auc = metrics.roc_auc_score(y_test-1, y_pre-1)
    print(score)
    return score,roc_auc

def rf_feature_prediction(data_version):
    X2 = same_length_csv('cap_feature.csv')
    nb_x_train = read_case_nb(f_in='nb_train_cv%d.dat' % (data_version))
    nb_x_test = read_case_nb(f_in='nb_test_cv%d.dat' % (data_version))
    y = same_length_csv('number_category.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X2, y, nb_x_train, nb_x_test, )
    lr = RandomForestClassifier(n_estimators=100,max_depth=3)
    lr = lr.fit(X_train, y_train, )
    score = lr.score(X_test, y_test)
    y_pre = lr.predict(X_test)
    roc_auc = metrics.roc_auc_score(y_test-1, y_pre-1)
    print(score)
    return score,roc_auc

def gbdt_merge_prediction(data_version):
    X = diff_length_csv_from1('temperature_37.2_40.0.csv')
    X = pad_sequences(X, maxlen=time_step, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    X2 = same_length_csv('cap_feature.csv')
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_cv%d.dat'%(data_version))
    y = same_length_csv('number_category.csv')
    X_con = np.concatenate((X2, X), axis=1)
    # print(X_con[0])
    X_train, X_test, y_train, y_test, = get_train_test_data(X_con, y, nb_x_train, nb_x_test, )
    lr = GradientBoostingClassifier(n_estimators=100,learning_rate=0.011,max_depth=3)
    lr = lr.fit(X_train, y_train, )
    # predicted =lr.predict(X_test)
    # f_out =open('gbdt_predicted.csv','a')
    # for i in range(0,len(nb_x_test)):
    #     f_out.write('%s,%d\n'%(nb_x_test[i],predicted[i]))
    score = lr.score(X_test, y_test)
    # print(lr.feature_importances_ )
    y_pre = lr.predict(X_test)
    roc_auc = metrics.roc_auc_score(y_test-1, y_pre-1)
    print(score)
    return score,roc_auc

def rf_merge_prediction(data_version):
    X = diff_length_csv_from1('temperature_37.2_40.0.csv')
    X = pad_sequences(X, maxlen=time_step, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    X2 = same_length_csv('cap_feature.csv')
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_cv%d.dat'%(data_version))
    y = same_length_csv('number_category.csv')
    X_con = np.concatenate((X2, X), axis=1)
    # print(X_con[0])
    X_train, X_test, y_train, y_test, = get_train_test_data(X_con, y, nb_x_train, nb_x_test, )
    lr = RandomForestClassifier(n_estimators=200,max_depth=3,)
    lr = lr.fit(X_train, y_train, )
    score = lr.score(X_test, y_test)
    # print(lr.feature_importances_)
    y_pre = lr.predict(X_test)
    roc_auc = metrics.roc_auc_score(y_test-1, y_pre-1)
    print(score)
    return score,roc_auc


def lr_avgTemp_prediction(data_version,penalty='l1'):
    X = diff_length_csv_from2('temperature_37.2_40.0.csv')
    new_x =np.zeros(shape=(len(X),2),)
    for k in range(0,len(X)):
        new_x[k][0]= sum(X[k])/len(X[k])
        # new_x[k][1]= max(X[k])
        # new_x[k][2] =min(X[k])

    X2 = same_length_csv('cap_feature.csv')
    print(X2.shape)
    nb_x_train =read_case_nb(f_in ='nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_cv%d.dat'%(data_version))
    y = same_length_csv('number_category.csv')
    X_con = np.concatenate((X2, new_x), axis=1)
    print(X_con.shape)
    print(X_con[0])
    X_train, X_test, y_train, y_test, = get_train_test_data(X_con, y, nb_x_train, nb_x_test, )
    gbdt = LogisticRegression(penalty=penalty, fit_intercept=True, max_iter=200, warm_start=True, tol=0.00005)
    gbdt = gbdt.fit(X_train, y_train, )
    score = gbdt.score(X_test, y_test)
    y_pre = gbdt.predict(X_test)
    roc_auc = metrics.roc_auc_score(y_test-1, y_pre-1)
    print(score)
    return score,roc_auc



def gbdt_avgTemp_prediction(data_version):
    X = diff_length_csv_from2('temperature_37.2_40.0.csv')
    new_x =np.zeros(shape=(len(X),2),)
    for k in range(0,len(X)):
        new_x[k][0]= sum(X[k])/len(X[k])
        new_x[k][1]= max(X[k])
        # new_x[k][2] =min(X[k])

    X2 = same_length_csv('cap_feature.csv')
    print(X2.shape)
    nb_x_train =read_case_nb(f_in ='nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_cv%d.dat'%(data_version))
    y = same_length_csv('number_category.csv')
    X_con = np.concatenate((X2, new_x), axis=1)
    print(X_con.shape)
    print(X_con[0])
    X_train, X_test, y_train, y_test, = get_train_test_data(X_con, y, nb_x_train, nb_x_test, )
    gbdt = GradientBoostingClassifier(n_estimators=100,learning_rate=0.011,max_depth=3)
    gbdt = gbdt.fit(X_train, y_train, )
    score = gbdt.score(X_test, y_test)
    y_pre = gbdt.predict(X_test)
    roc_auc = metrics.roc_auc_score(y_test-1, y_pre-1)
    print(score)
    return score,roc_auc

def print_score():
    score = 0.0
    auc =0.0
    for data_version in range(1,6):
        # rt_acc, rt_auc = logistic_regression_temperature_prediction(data_version=data_version, penalty='l1')
        # rt_acc, rt_auc = logistic_regression_temperature_prediction(data_version=data_version,penalty='l2')
        # rt_acc, rt_auc = logistic_regression_feature_prediction(data_version=data_version,penalty='l1')
        # rt_acc, rt_auc = logistic_regression_feature_prediction(data_version=data_version,penalty='l2')
        # rt_acc, rt_auc = logistic_regression_prediction(data_version=data_version,penalty='l1')
        rt_acc, rt_auc = logistic_regression_prediction(data_version=data_version,penalty='l2')
        # rt_acc, rt_auc = gbdt_temprature_prediction(data_version=data_version,)
        # rt_acc, rt_auc = gbdt_feature_prediction(data_version=data_version,)
        # rt_acc, rt_auc = gbdt_merge_prediction(data_version=data_version,)
        # rt_acc, rt_auc = rf_merge_prediction(data_version=data_version,)
        # rt_acc, rt_auc = rf_feature_prediction(data_version=data_version,)
        # rt_acc, rt_auc = rf_temprature_prediction(data_version=data_version,)

        # rt_acc,rt_auc =gbdt_avgTemp_prediction(data_version)
        # rt_acc,rt_auc =lr_avgTemp_prediction(data_version,penalty='l2')
        score += rt_acc
        auc+=rt_auc
    print('score',score /5)
    print('auc',auc /5)
    return score/5,auc/5

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


def predict_accuracy(f_in ='gbdt_predicted.csv',f_truth ='number_category.csv'):
    f_in =open(f_in)
    predicted=[]
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        predicted.append(line)
    f_truth =open(f_truth)
    number=[]
    truth=[]
    for i, lines in enumerate(f_truth):
        line =lines.strip().split(',')[0:]
        number.append(line[0])
        truth.append(line[1])

    count=0.0
    true_case_1 =[]
    true_case_2 =[]
    false_case_1=[]
    false_case_2=[]
    for k in range(0,len(predicted)):
        if predicted[k][0] in number:
            index =number.index(predicted[k][0])
            if predicted[k][1] ==truth[index]:
                count+=1
                if predicted[k][1]=='1':
                    true_case_1.append(predicted[k][0])
                elif predicted[k][1]=='2':
                    true_case_2.append(predicted[k][0])
            else:
                print(predicted[k][0])
                if predicted[k][1]=='1':
                    false_case_1.append(predicted[k][0])
                elif predicted[k][1]=='2':
                    false_case_2.append(predicted[k][0])

    acc =count/len(predicted)
    print(acc)
    return true_case_1,true_case_2,false_case_1,false_case_2


def plot_pca(f_in='gbdt_predicted.csv'):
    X = diff_length_csv('temperature.csv')
    X = pad_sequences(X, maxlen=1, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    X2 = same_length_csv('cap_feature.csv')
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_cv%d.dat'%(data_version))
    y = same_length_csv('number_category.csv')
    X_con = np.concatenate((X2, X), axis=1)
    nb =read_case_nb(f_in='number.csv')

    true_case_1,true_case_2,false_case_1,false_case_2 =predict_accuracy(f_in=f_in)

    case =np.concatenate([true_case_1,true_case_2,false_case_1,false_case_2],axis=0)
    y_true =[]
    for i in range(0,len(true_case_1)+len(true_case_2)):
        y_true.append([2])
    y_false =[]
    for i in range(0,len(false_case_1)+len(false_case_2)):
        y_false.append([1])

    y_predict =np.concatenate([y_true,y_false],axis=0)

    X_train, X_test, y_train, y_test, = get_train_test_data(X_con, y, nb, nb_x_test, )
    X_true, X_false, y_true, y_false, = get_train_test_data(X_con, y_predict, case, false_case_1, )

    # print(y_true)
    X_reduced = PCA(n_components=2).fit_transform(X_train)
    X_true = PCA(n_components=2).fit_transform(X_true)

    for i in range(0,len(X_reduced)):
        if X_reduced[i,0]>15000 and y[i][1]==2:
            print('2*',nb[i])
        if X_reduced[i,0]>15000 and  y[i][1]==1:
            print('1*',nb[i])




    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_train,edgecolors='k',s=20)
    plt.title('all_data')
    figure()
    plt.scatter(X_true[0:len(true_case_1),0],X_true[0:len(true_case_1),1],c='r',edgecolors='k',s=30,label='true_case_1')
    plt.scatter(X_true[len(true_case_1):len(true_case_1)+len(true_case_2),0],X_true[len(true_case_1):len(true_case_1)+len(true_case_2),1],c='b',edgecolors='k',s=30,label='true_case_2')
    plt.scatter(X_true[len(true_case_1)+len(true_case_2):len(true_case_1)+len(true_case_2)+len(false_case_1),0],X_true[len(true_case_1)+len(true_case_2):len(true_case_1)+len(true_case_2)+len(false_case_1),1],c='g',edgecolors='k',s=30,label='false_case_1')
    plt.scatter(X_true[len(true_case_1)+len(true_case_2)+len(false_case_1):,0],X_true[len(true_case_1)+len(true_case_2)+len(false_case_1):,1],c='black',edgecolors='k',s=30,label='false_case_2')
    plt.title('gbdt_feature_predicted')
    # figure()
    # plt.scatter(X_false[:,0],X_false[:,1],c=cValue,s=40,edgecolors='k')
    # plt.title('gbdt_false_predict')
    legend()
    show()


def temp_length_study(input):

    f_in =open('unpadding_temp.csv')
    nb=[]
    rt_list=[]
    for i ,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        nb.append(line[0])
        rt_list.append(line[1:])
    count=0.0
    length_list=[]
    print(type(nb[0]))
    for i in range(0,len(input)):
        if str(float(input[i])) in nb:
            index =nb.index(str(float(input[i])))
            length_list.append(len(rt_list[index]))
            count+=len(rt_list[index])
    length_count =0.0
    for k in range(0,len(length_list)):
        if length_list[k]<=5:
            length_count+=1

    print('length less than 10',length_count)
    print('count',count)
    print('average length',count/len(input))


def case_study():
    true_case_1_1,true_case_2_1,false_case_1_1,false_case_2_1 =predict_accuracy(f_in='gbdt_predicted.csv')
    true_case_1_2, true_case_2_2, false_case_1_2, false_case_2_2 =predict_accuracy(f_in='lr_l2_predicted.csv')
    true_case_1_3, true_case_2_3, false_case_1_3, false_case_2_3 =predict_accuracy(f_in='merge_att_3_lstm_5_predicted.csv')
    true_case_1_4, true_case_2_4, false_case_1_4, false_case_2_4 =predict_accuracy(f_in='gbdt_feature_predicted.csv')


    print('false_case_1_1',len(false_case_1_1))  #70
    print('false_case_2_1',len(false_case_2_1))  #78
    print('false_case_1_2',len(false_case_1_2))  #80
    print('false_case_2_2',len(false_case_2_2))  #106
    print('false_case_1_3',len(false_case_1_3))  #72
    print('false_case_2_3',len(false_case_2_3))  #116
    print('intersection false_case_1 1 2',len(set.intersection(set(false_case_1_1),set(false_case_1_2))),set.intersection(set(false_case_1_1),set(false_case_1_2)),)
    print('intersection false_case_1 1 4',len(set.intersection(set(false_case_1_1),set(false_case_1_4))),set.intersection(set(false_case_1_1),set(false_case_1_4)),)
    print('diff true_case_1 1 4',len(set(true_case_1_1).difference(set(true_case_1_4))),set(true_case_1_1).difference(set(true_case_1_4)))
    print('diff true_case_1 1 4',len(set(true_case_1_4).difference(set(true_case_1_1))),set(true_case_1_4).difference(set(true_case_1_1)))
    print('intersection false_case_1 2 3',len(set.intersection(set(false_case_1_2),set(false_case_1_3))),set.intersection(set(false_case_1_2),set(false_case_1_3)),)
    print('intersection false_case_1 1 3',len(set.intersection(set(false_case_1_1),set(false_case_1_3))),set.intersection(set(false_case_1_1),set(false_case_1_3)),)
    print('intersection false_case_1 1 2 3',len(set.intersection(set(false_case_1_1),set(false_case_1_2),set(false_case_1_3))),set.intersection(set(false_case_1_1),set(false_case_1_2),set(false_case_1_3)),)

    print('intersection false_case_2 1 2',len(set.intersection(set(false_case_2_1),set(false_case_2_2))),set.intersection(set(false_case_2_1),set(false_case_2_2)),)
    print('intersection false_case_2 1 4',len(set.intersection(set(false_case_2_1),set(false_case_2_4))),set.intersection(set(false_case_2_1),set(false_case_2_4)),)
    print('diff true_case_1 1 4', len(set(true_case_2_1).difference(set(true_case_2_4))),set(true_case_2_1).difference(set(true_case_2_4)))
    print('diff true_case_1 1 4', len(set(true_case_2_4).difference(set(true_case_2_1))),set(true_case_2_4).difference(set(true_case_2_1)))
    print('intersection false_case_2 2 3',len(set.intersection(set(false_case_2_2),set(false_case_2_3))),set.intersection(set(false_case_2_2),set(false_case_2_3)),)
    print('intersection false_case_2 1 3',len(set.intersection(set(false_case_2_1),set(false_case_2_3))),set.intersection(set(false_case_2_1),set(false_case_2_3)),)
    print('intersection false_case_2 1 2 3',len(set.intersection(set(false_case_2_1),set(false_case_2_2),set(false_case_2_3))),set.intersection(set(false_case_2_1),set(false_case_2_2),set(false_case_2_3)),)



    # false_case_1_123 =list(set.intersection(set(false_case_1_1), set(false_case_1_2), set(false_case_1_3)))
    # false_case_2_123 =list(set.intersection(set(false_case_2_1), set(false_case_2_2), set(false_case_2_3)))
    # temp_length(false_case_1_123)
    # temp_length(false_case_2_123)
    # insect_123 =np.concatenate((false_case_1_123,false_case_2_123),axis=0)
    # temp_length(insect_123)
    # all_data =read_case_nb(f_in='number.csv')
    # temp_length(all_data)


def padding_temp(f_in ='temperature.csv',f_out='unpadding_temp.csv'):
    f_in =open(f_in)
    rt_list=[]
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        line =[float(t) for t in line]
        for k in range(0,len(line)):
            if sum(line[len(line)-k-1:len(line)])!=0:
                rt_list.append(line[0:len(line)-k])
                break
    f_out=open(f_out,'w')
    for i in range(0,len(rt_list)):
        for j in range(0,len(rt_list[i])):
            if j ==0:
                f_out.write('%.1f'%(rt_list[i][0]))
            else:
                f_out.write(',%.3f'%(rt_list[i][j]))
        f_out.write('\n')

def temp_period(f_in='temperature_raw.csv',period_min =37.2,period_max =38.5):
    f_in =open(f_in)
    rt_list=[]
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[1:]
        line =[float(k) for k in line]
        rt_list.append(line)
    count =0.0
    for i in range(0,len(rt_list)):
        for j in range(0,len(rt_list[i])):
            if period_min<=rt_list[i][j]<=period_max:
                count+=1
    print(count)

def normalization_solution_2(f_in='temperature_raw.csv',f_out='temperature.csv',period_min =37.2,period_max =38.5):
    f_in=open(f_in)
    f_out=open(f_out,'w')
    for i ,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        number =line[0]
        temp =[float(v) for v in line[1:]]
        f_out.write(number)

        for i in range(0,len(temp)):
            if temp[i]<=37.2:
                f_out.write(',0.000')
            elif temp[i]>=38.5:
                f_out.write(',1.000')
            elif 37.2<temp[i]<38.5:
                f_out.write(',%.3f'%((temp[i]-period_min)/(period_max-period_min)))
        f_out.write('\n')


def age(f_in ='number_age.csv',f_nb='number.csv',f_out='age.csv'):
    f_in =open(f_in)
    f_nb=open(f_nb)
    f_out =open(f_out,'w')
    number=[]
    for i ,lines in enumerate(f_nb):
        line =lines.strip()
        number.append(line)
    nb =[]
    age=[]
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        nb.append(line[0])
        age.append(line[1])
    for i in range(0,len(nb)):
        if nb[i] in number:
            f_out.write('%s,%s\n'%(nb[i],age[i]))

def describe_col(f_in='age.csv',col_nb =1):
    f_in =open(f_in)
    col =[]
    for i ,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        col.append(float(line[col_nb]))

    print('average',np.mean(np.array(col)))
    print('max',np.max(np.array(col)))
    print('min',np.min(np.array(col)))


def category(f_in ='number_category.csv'):
    f_in =open(f_in)
    rt_list=[]
    for i ,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        rt_list.append(line)
    count=[0,0]
    for i in range(0,len(rt_list)):
        if rt_list[i][1]=='1':
            count[0]+=1
        elif rt_list[i][1]=='2':
            count[1]+=1
    print(count)

def category_label(f_in='category_3.csv',label=3):
    f_in=open(f_in)
    rt_list=[]
    for i,lines in enumerate(f_in):
        line =lines.strip()[0]
        rt_list.append(line)
    count=np.zeros(shape=(1,label))
    for i in range(0,len(rt_list)):

        count[0][int(rt_list[i])-1]+=1
    print(count)

def temp_length(f_in='temperature_37.2_40.0.csv'):
    f_in =open(f_in)
    rt_list=np.zeros(shape=(150))
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[1:]
        rt_list[len(line)]+=1
    print(rt_list)
    print(sum(rt_list[0:60]))
    f_out= open('temp_length.dat','w')
    for i in range(0,150):
        if i==0:
            f_out.write('number length\n')
            continue
        f_out.write('%d %.4f\n'%(i,sum(rt_list[0:i+1]/699)))
        if sum(rt_list[0:i]) == 699:
            exit()
    return rt_list

def plot_heatmap():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    f_nb=open('nb_test_cv5.dat')
    nb_list=[]
    for i ,lines in enumerate(f_nb):
        line=lines.strip().split(' ')[0]
        nb_list.append(line)
    print(nb_list)
    f_y=open('number_category.csv')
    nb_ca_list=[]
    for i,lines in enumerate(f_y):
        line =lines.strip().split(',')[0:]
        nb_ca_list.append(line)
    print(nb_ca_list)
    category=[]
    for m in range(0,len(nb_list)):
        for n in range(0,len(nb_ca_list)):
            if int(nb_list[m])==int(nb_ca_list[n][0]):
                category.append(nb_ca_list[n][1])
                break
    print('category',category)

    f_in=open('heatmap.dat')
    cate_1=[]
    cate_2=[]
    for i,lines in enumerate(f_in):
        line =lines.strip().split(' ')[1:]
        line =[float(l) for l in line]
        if category[i] == '1':
            cate_1.append(line)
        if category[i] =='2':
            cate_2.append(line)
    print(cate_2)
    print(np.array(cate_2).shape)
    data =np.concatenate([cate_1,cate_2],axis=0)

    print(np.array(data).shape)


    # remove index title
    # nba.index.name = ""
    # nba =nba.drop('0',0)

    # normalize data columns

    # nba_norm = (nba - nba.mean()) / (nba.max() - nba.min())
    # relabel columns

    # nba_norm.columns = labels
    # set appropriate font and dpi

    sns.set(font_scale=1.2)
    sns.set_style({"savefig.dpi": 100})
    # plot it out
    ax = sns.heatmap(data, cmap=plt.cm.Blues, linewidths=.1)
    # set the x-axis labels on the top
    ax.xaxis.tick_top()
    plt.xlabel('temp position')
    plt.ylabel('number of patient')
    # rotate the x-axis labels
    # plt.xticks(rotation=90)
    # get figure (usually obtained via "fig,ax=plt.subplots()" with matplotlib)
    fig = ax.get_figure()
    # specify dimensions and save
    fig.set_size_inches(15, 20)
    show()
    # fig.savefig("nba.png")

def func(f_in ='heatmap_2.dat'):
    f_in =open(f_in,'a')
    f_in.write('0')
    for i in range(0,140):
        f_in.write(' %d'%(i))


def Normalization_self(f_in='temperature_raw.csv',f_out='temperature_37.2_40.0.csv',period_min =37.2,period_max=40.0):
    f_in =open(f_in)
    f_out=open(f_out,'w')
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        line =[float(v) for v in line]
        f_out.write('%.1f'%(line[0]))
        for k in range(1,len(line[0:])):
            if line[k]>=period_max:
                f_out.write(',%.7f'%(1))
            elif line[k]<=period_min:
                f_out.write(',%.7f'%(0))
            else:
                f_out.write(',%.7f'%((line[k]-period_min)/(period_max-period_min)))
        f_out.write('\n')




if __name__ =='__main__':
    print('main')
    time_step = 50
    # Normalization_self()
    # exit()
    # plot_heatmap()
    # temp_length()
    # describe_col()
    # category()
    # category_label(f_in='category_3.csv',label=3)
    # category_label(f_in='category_5.csv',label=5)
    # category_split(category=1)
    # category_split(category=2)
    # nb_5_cv_split(nb_cv=1)
    # nb_5_cv_split(nb_cv=2)
    # nb_5_cv_split(nb_cv=3)
    # nb_5_cv_split(nb_cv=4)
    # nb_5_cv_split(nb_cv=5)
    # padding_temp()
    # normalization_solution_2(f_out='temperature.csv',period_max=38.5)


    count=0.0
    auc=0.0
    for i in range(1,6):
        acc,auc_ =print_score()
        count+=acc
        auc+=auc_
    print('acc',count/10)
    print('auc',auc/10)
    print(time_step)

    # print_score()
    # print(time_step)
    # case_study()
    # temp_period(period_min=37.2,period_max=40.0)


    # plot_pca(f_in='merge_att_3_lstm_5_predicted.csv')
    # plot_pca(f_in='gbdt_feature_predicted.csv')
    # for data_version in range(2,3):
        # logistic_regression_prediction_epoch(data_version=data_version,penalty='l2')
        # logistic_regression_temprature_prediction_epoch(data_version=data_version,penalty='l2')
        # ylim((0.5,1.0))
        # show()





