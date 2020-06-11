from accessory import same_length_csv,diff_length_csv,read_case_nb,get_train_test_data
import numpy as np
from matplotlib.pyplot import plot,legend,show,title,figure,ylim
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from keras.preprocessing.sequence import pad_sequences
np.random.seed(1337)
nb_classes =2
data_version =5
nb_epochs=200
batch_size =16

def temperature_length_longer_than_12(f_in ='temperature.csv',length=14):
    f_in =open(f_in)
    rt_list=[]
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        if len(line)>=length:
            rt_list.append(line)
    f_out =open('temperature_lt%d.csv'%(length-2),'w')
    for i in range(0,len(rt_list)):
        for j in range(0,len(rt_list[i])):
            if j ==0:
                f_out.write(rt_list[i][0])
            else:
                f_out.write(',%s'%(rt_list[i][j]))
        f_out.write('\n')

# temperature_length_longer_than_12()
# temperature_length_longer_than_12(length=20)

def get_number(f_in ='temperature_lt12.csv'):
    f_in =open(f_in)
    f_out =open('number_lt12.csv','w')
    for i ,lines in enumerate(f_in):
        line =lines.strip().split(',')[0]
        f_out.write('%s\n'%line)

# get_number()

def nb_5_cv_split(f_in ='number_lt12.csv',nb_cv =1):
    f_in =open(f_in)
    f_train_cv1 =open('nb_train_lt12_cv%d.dat'%(nb_cv),'w')
    f_test_cv1 =open('nb_test_lt12_cv%d.dat'%(nb_cv),'w')
    for i,lines in enumerate(f_in):
        line =lines.strip()
        t =i%5
        if t !=nb_cv-1:
            f_train_cv1.write(line+'\n')
        else:
            f_test_cv1.write(line+'\n')

# nb_5_cv_split(nb_cv=1)
# nb_5_cv_split(nb_cv=2)
# nb_5_cv_split(nb_cv=3)
# nb_5_cv_split(nb_cv=4)
# nb_5_cv_split(nb_cv=5)

def get_lt12(f_in_1='number_lt12.csv',f_in_2='cap_feature.csv',f_out ='cap_feature_lt12.csv'):
    f_in_1 =open(f_in_1)
    f_in_2 =open(f_in_2)
    f_out =open(f_out,'w')
    number =[]
    for i ,lines in enumerate(f_in_1):
        line =lines.strip().split(',')[0]
        number.append(float(line))
    rt_list=[]
    for i ,lines in enumerate(f_in_2):
        line =lines.strip().split(',')[0:]
        if float(line[0]) in number:
            rt_list.append(line)
    for i in range(0,len(rt_list)):
        for j in range(0,len(rt_list[i])):
            if j ==0:
                f_out.write(rt_list[i][0])
            else:
                f_out.write(',%s'%(rt_list[i][j]))
        f_out.write('\n')

# get_lt12()
# get_lt12(f_in_2='number_category.csv',f_out='number_category_lt12.csv')

def logistic_regression_prediction_lt12(data_version,penalty='l1'):
    X = diff_length_csv('temperature_lt12.csv')
    X = pad_sequences(X, maxlen=50, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    X2 = same_length_csv('cap_feature_lt12.csv')
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_lt12_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_lt12_cv%d.dat'%(data_version))
    y = same_length_csv('number_category_lt12.csv')
    X_con = np.concatenate((X2, X), axis=1)
    X_train, X_test, y_train, y_test, = get_train_test_data(X_con, y, nb_x_train, nb_x_test, )
    lr =LogisticRegression(penalty=penalty,fit_intercept=True,max_iter=200,tol=0.00005)
    lr =lr.fit(X_train,y_train,)
    # print(lr.coef_)
    score =lr.score(X_test,y_test)
    print(score)
    return score

def logistic_regression_prediction_epoch_lt12(data_version,penalty='l1'):
    X = diff_length_csv('temperature_lt12.csv')
    X = pad_sequences(X, maxlen=50, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    X2 = same_length_csv('cap_feature_lt12.csv')
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_lt12_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_lt12_cv%d.dat'%(data_version))
    y = same_length_csv('number_category_lt12.csv')
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

def logistic_regression_feature_prediction_lt12(data_version,penalty ='l1'):
    X2 = same_length_csv('cap_feature_lt12.csv')
    nb_x_train =read_case_nb(f_in ='nb_train_lt12_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_lt12_cv%d.dat'%(data_version))
    y = same_length_csv('number_category_lt12.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X2, y, nb_x_train, nb_x_test, )
    lr = LogisticRegression(penalty=penalty, fit_intercept=True, max_iter=200, warm_start=True,tol=0.0001)
    lr = lr.fit(X_train, y_train, )
    # print(lr.coef_)
    score = lr.score(X_test, y_test)
    print(score)
    return score

def logistic_regression_temperature_prediction_lt12(data_version,penalty ='l1'):
    X = diff_length_csv('temperature_lt12.csv')
    X = pad_sequences(X, maxlen=50, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_lt12_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_lt12_cv%d.dat'%(data_version))
    y = same_length_csv('number_category_lt12.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    lr = LogisticRegression(penalty=penalty, fit_intercept=True, max_iter=200, warm_start=True)
    lr = lr.fit(X_train, y_train, )
    # print(lr.coef_)
    score = lr.score(X_test, y_test)
    print(score)
    return score

def logistic_regression_temprature_prediction_epoch_lt12(data_version,penalty ='l1'):
    X = diff_length_csv('temperature_lt12.csv')
    X = pad_sequences(X, maxlen=50, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_lt12_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_lt12_cv%d.dat'%(data_version))
    y = same_length_csv('number_category_lt12.csv')
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

def gbdt_temprature_prediction_lt12(data_version):
    X = diff_length_csv('temperature_lt12.csv')
    X = pad_sequences(X, maxlen=50, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_lt12_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_lt12_cv%d.dat'%(data_version))
    y = same_length_csv('number_category_lt12.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )

    lr = GradientBoostingClassifier(n_estimators=100,learning_rate=0.01,max_depth=3)
    lr = lr.fit(X_train, y_train, )
    score = lr.score(X_test, y_test)
    print(score)
    return score

def gbdt_feature_prediction_lt12(data_version):
    X2 = same_length_csv('cap_feature_lt12.csv')
    nb_x_train = read_case_nb(f_in='nb_train_lt12_cv%d.dat' % (data_version))
    nb_x_test = read_case_nb(f_in='nb_test_lt12_cv%d.dat' % (data_version))
    y = same_length_csv('number_category_lt12.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X2, y, nb_x_train, nb_x_test, )
    lr = GradientBoostingClassifier(n_estimators=100,learning_rate=0.01,max_depth=3)
    lr = lr.fit(X_train, y_train, )
    score = lr.score(X_test, y_test)
    print(score)
    return score


def gbdt_merge_prediction_lt12(data_version):
    X = diff_length_csv('temperature_lt12.csv')
    X = pad_sequences(X, maxlen=50, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    X2 = same_length_csv('cap_feature_lt12.csv')
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_lt12_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_lt12_cv%d.dat'%(data_version))
    y = same_length_csv('number_category_lt12.csv')
    X_con = np.concatenate((X2, X), axis=1)
    X_train, X_test, y_train, y_test, = get_train_test_data(X_con, y, nb_x_train, nb_x_test, )
    lr = GradientBoostingClassifier(n_estimators=100,learning_rate=0.01,max_depth=3)
    lr = lr.fit(X_train, y_train, )
    score = lr.score(X_test, y_test)
    print(score)
    return score

def print_score_lt12():
    score = 0.0
    for data_version in range(1, 6):
        # score += logistic_regression_temperature_prediction_lt12(data_version=data_version, penalty='l1')
        # score += logistic_regression_temperature_prediction_lt12(data_version=data_version,penalty='l2')
        # score += logistic_regression_feature_prediction_lt12(data_version=data_version,penalty='l1')
        # score += logistic_regression_feature_prediction_lt12(data_version=data_version,penalty='l2')
        # score += logistic_regression_prediction_lt12(data_version=data_version,penalty='l1')
        # score += logistic_regression_prediction_lt12(data_version=data_version,penalty='l2')
        # score += gbdt_temprature_prediction_lt12(data_version=data_version,)
        # score += gbdt_feature_prediction_lt12(data_version=data_version)
        score += gbdt_merge_prediction_lt12(data_version=data_version,)
    print(score /5)

print_score_lt12()