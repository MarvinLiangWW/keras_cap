import csv
from datetime import datetime,timedelta
import numpy as np
import random
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from keras.layers import Dropout,Dense,Activation
# from matplotlib.pyplot import savefig,plot,legend,show,title
from keras.optimizers import Adam,SGD
from sklearn.linear_model import LogisticRegression,LinearRegression
from keras.preprocessing.sequence import pad_sequences
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

def fun3(f_in='nb_x_train.dat'):
    f_in=open(f_in)
    nb_x=[]
    for i,lines in enumerate(f_in):
        line =lines.strip()
        nb_x.append(line)
    return nb_x

def get_train_validation_test_data(X_con,y,nb_x_train,nb_x_test,nb_x_validation):
    X_train,X_test,X_validation,y_train,y_test,y_validation=[],[],[],[],[],[]
    for m in range(0,len(nb_x_train)):
        for n in range(0,len(X_con)):
            if float(nb_x_train[m])==float(X_con[n][0]):
                X_train.append(X_con[n][1:])
                break
    for m in range(0,len(nb_x_train)):
        for n in range(0,len(y)):
            if float(nb_x_train[m])==float(y[n][0]):
                y_train.append(y[n][1:])
                break
    for m in range(0,len(nb_x_test)):
        for n in range(0,len(X_con)):
            if float(nb_x_test[m])==float(X_con[n][0]):
                X_test.append(X_con[n][1:])
                break
    for m in range(0,len(nb_x_test)):
        for n in range(0,len(y)):
            if float(nb_x_test[m])==float(y[n][0]):
                y_test.append(y[n][1:])
                break
    for m in range(0,len(nb_x_validation)):
        for n in range(0,len(X_con)):
            if float(nb_x_validation[m])==float(X_con[n][0]):
                X_validation.append(X_con[n][1:])
                break
    for m in range(0,len(nb_x_validation)):
        for n in range(0,len(y)):
            if float(nb_x_validation[m])==float(y[n][0]):
                y_validation.append(y[n][1:])
                break
    return np.array(X_train),np.array(X_test),np.array(X_validation),np.array(y_train),np.array(y_test),np.array(y_validation)

def reshape_dataset(train):
    trainX=np.reshape(train,(train.shape[0],train.shape[1],1))
    return np.array(trainX)

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


def fun2(filename):
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



def split_train_validation_test_data():
    f_in =open('number.csv')
    number=[]
    for i,lines in enumerate(f_in):
        line =lines.strip()
        number.append(line)
    a= [i for i in range(0,len(number))]
    for k in range(1,6):
        random.shuffle(a)
        f_out_train=open('nb_train_'+str(k)+'.dat','w')
        f_out_test=open('nb_test_'+str(k)+'.dat','w')
        f_out_validation=open('nb_validation_'+str(k)+'.dat','w')
        for l in range(0,len(number)):
            if l<800:
                f_out_train.write('%.1f\n'%float(number[a[l]]))
            elif 800<=l<1000:
                f_out_validation.write('%.1f\n'%float(number[a[l]]))
            else:
                f_out_test.write('%.1f\n'%float(number[a[l]]))


def split_train_test():
    f_in = open('number.csv')
    number = []
    for i, lines in enumerate(f_in):
        line = lines.strip()
        number.append(line)
    a = [i for i in range(0, len(number))]
    for k in range(1, 6):
        random.shuffle(a)
        f_out_train = open('nb_x_train_' + str(k) + '.dat', 'w')
        f_out_test = open('nb_x_test_' + str(k) + '.dat', 'w')
        for l in range(0, len(number)):
            if l < 500:
                f_out_train.write('%.1f\n' % float(number[a[l]]))
            else:
                f_out_test.write('%.1f\n' % float(number[a[l]]))

# split_train_test()

def percentage(f_in ='number_category.csv'):
    f_in = open(f_in)
    category=[]
    count =[]
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        if line[1] in category:
            li =category.index(line[1])
            count[li]+=1
        else:
            category.append(line[1])
            count.append(0)
    sum_count =sum(count)
    for i in range(0 ,len(category)):
        print('%s %d %.3f'%(category[i],count[i],count[i]/sum_count))

# percentage(f_in='number_category.csv')

def split_number_category_tofile(f_in ='number_category.csv'):
    f_in =open(f_in)
    f_out_number=open('number.csv','w')
    f_out_category=open('category.csv','w')
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        f_out_number.write(line[0]+'\n')
        f_out_category.write(line[1]+'\n')

# split_number_category_tofile()

def normalization_solution_1(f_in ='5_day_25_check.csv',f_out='5_day_25_nor.dat'):
    f_in= open(f_in)
    f_out =open(f_out,'w')
    for i ,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        number =line[0]
        temp =line[1:]
        max_temp =float(max(temp))
        min_temp =float(min(temp))
        f_out.write(number)
        if max_temp == min_temp:
            for temp_i in temp:
                f_out.write(',1.000')
        else:
            for temp_i in temp:
                f_out.write(',%.3f'%((float(temp_i)-min_temp)/(max_temp-min_temp)))
        f_out.write('\n')

# normalization_solution_1(f_in ='2_5_day_tiwencheck.csv',f_out='2_5_day_nor_s1.csv')

def normalization_solution_2(f_in='2_5_day_tiwencheck.csv',f_out='2_5_day_nor_s2.csv',standard_temp =37.2):
    f_in=open(f_in)
    f_out=open(f_out,'w')
    for i ,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        number =line[0]
        temp =line[1:]
        max_temp=float(max(temp))
        f_out.write(number)
        if max_temp == standard_temp:
            for temp_i in temp:
                f_out.write(',1.000')
        else:
            for temp_i in temp:
                if float(temp_i)<=standard_temp:
                    f_out.write(',0.000')
                else:
                    f_out.write(',%.3f'%((float(temp_i)-standard_temp)/(max_temp-standard_temp)))
        f_out.write('\n')

# normalization_solution_2(f_in='2_5_day_tiwencheck.csv',f_out='2_5_day_nor_s2.csv')

def get_new_number_category(f_in_1 ='2_5_day_tiwencheck.csv',f_in_2='number_category.csv',f_out='number_category_2.csv'):
    f_in_1 =open(f_in_1)
    number =[]
    for i,lines in enumerate(f_in_1):
        number.append(lines.strip().split(',')[0])
    f_in_2=open(f_in_2)
    number_category=[]
    for i,lines in enumerate(f_in_2):
        number_category.append(lines.strip().split(',')[0:])
    f_out =open(f_out,'w')
    for m in range(0,len(number)):
        for n in range(0,len(number_category)):
            if float(number[m])==float(number_category[n][0]):
                f_out.write(number_category[n][0])
                for k in range(1,len(number_category[n])):
                    f_out.write(','+number_category[n][k])
        f_out.write('\n')

# get_new_number_category(f_in_1='number.csv',f_in_2='number_age_col85tran.csv',f_out='number_age_col85tran_v2.csv')
# get_new_number_category(f_in_1='number.csv',f_in_2='number_age_col71tran.csv',f_out='number_age_col71tran_v2.csv')

def get_templength():
    f_in= open('2_5_day_nor_s2.csv')
    f_out =open('5_day_temp_length.csv','w')
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        f_out.write('%.1f,%d\n'%(float(line[0]),len(line)-1))

# get_templength()

def length_prediction():
    X = fun2('5_day_temp_length_s1.csv')
    y = fun2('number_category.csv')
    nb_x_train = fun3(f_in='nb_x_train_%d.dat' % data_version)
    nb_x_test = fun3(f_in='nb_x_test_%d.dat' % data_version)
    x_train, x_test, y_train, y_test = get_train_test_data(X, y, nb_x_train, nb_x_test, )

    probability_test = (sum(y_test) - len(y_test)) / len(y_test)
    print('probability_test:', probability_test)
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    model =Sequential()
    model.add(Dense(2,input_shape=(1,)))
    model.add(Dropout(0.1))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001,),
                  metrics=['accuracy'])
    model.summary()
    # return model

    print('Train...')
    acc_list = []
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    acc_list.append(acc)
    for epoch in range(nb_epochs):
        # print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size,verbose=1, nb_epoch=1, validation_split=0.05)
        score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        acc_list.append(acc)

        print('Test score:', score)
        print('Test accuracy:', acc)
    plot(range(0, nb_epochs + 1), [probability_test for i in range(0, nb_epochs + 1)])
    plot(range(0, nb_epochs + 1), acc_list)
    show()


# length_prediction()

def normalization_solution_3(f_in='5_day_temp_length.csv',f_out='5_day_temp_length_s1.csv'):
    f_in =open(f_in)
    f_out=open(f_out,'w')
    number =[]
    number_length=[]
    for i ,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        number.append(line[0])
        number_length.append(line[1])
    max_length =float(max(number_length))
    min_length =float(min(number_length))
    for i in range(0,len(number)):
        f_out.write(number[i]+',')
        f_out.write(('%.7f')%((float(number_length[i])-min_length)/(max_length-min_length)))
        f_out.write('\n')

# normalization_solution_3()


def combine_fun(f_in_1='number_age_col71tran.csv',f_in_2='number_daye.dat',f_out='number_age_col73tran.csv'):
    f_in_1 = open(f_in_1)
    f_in_2=open(f_in_2)
    metrix_1 =[]
    metrix_2=[]
    for i,lines in enumerate(f_in_1):
        line =lines.strip().split(',')[0:]
        metrix_1.append(line)
    for k,lines in enumerate(f_in_2):
        line =lines.strip().split(',')[0:]
        metrix_2.append(line)
    for j in range(0,len(metrix_1)):
        for l in range(0,len(metrix_2)):
            if int(float(metrix_1[j][0]))==float(metrix_2[l][0]):
                for m in range(1,len(metrix_2[l])):
                    metrix_1[j].append(metrix_2[l][m])
                break
    f_out=open(f_out,'w')
    for i in range(0,len(metrix_1)):
        f_out.write('%.1f'%(float(metrix_1[i][0])))
        for j in range(1,len(metrix_1[i])):
            f_out.write(',%.7f'%(float(metrix_1[i][j])))
        f_out.write('\n')


# combine_fun(f_in_1='number_age_col85tran_v2.csv',f_in_2='5_day_temp_length_s1.csv',f_out='number_age_col86tran_v2.csv')

def logistic_regression_prediction(data_version):
    X = diff_length_csv('2_5_day_nor_s2.csv')
    X = pad_sequences(X, maxlen=20, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    # X2 = fun2('number_age_col85tran_v2.csv')
    # X2 = fun2('number_age_col71tran_v2.csv')
    X2 = fun2('cap_feature_2.csv')
    nb_x_train = fun3(f_in='nb_x_train_%d.dat' % (data_version))
    nb_x_test = fun3(f_in='nb_x_test_%d.dat' % (data_version))
    y = fun2('number_category.csv')
    X_con = np.concatenate((X2, X), axis=1)
    X_train, X_test, y_train, y_test, = get_train_test_data(X_con, y, nb_x_train, nb_x_test, )
    lr =LogisticRegression(penalty='l2',fit_intercept=True,max_iter=200,warm_start=True)
    # lr =LinearRegression()
    # lr =LogisticRegression(penalty='l2')
    lr =lr.fit(X_train,y_train,)
    print(lr.coef_)
    score =lr.score(X_test,y_test)
    print(score)
    return score

def logistic_regression_feature_prediction(data_version):
    # X2 = fun2('number_age_col85tran_v2.csv')
    # X2 = fun2('number_age_col71tran_v2.csv')
    X2 = fun2('cap_feature_2.csv')
    nb_x_train = fun3(f_in='nb_x_train_%d.dat' % (data_version))
    nb_x_test = fun3(f_in='nb_x_test_%d.dat' % (data_version))
    y = fun2('number_category.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X2, y, nb_x_train, nb_x_test, )
    lr = LogisticRegression(penalty='l1', fit_intercept=True, max_iter=200, warm_start=True)
    # lr =LinearRegression()
    # lr =LogisticRegression(penalty='l2')
    lr = lr.fit(X_train, y_train, )
    # print(lr.coef_)
    score = lr.score(X_test, y_test)
    print(score)
    return score

def logistic_regression_temperature_prediction(data_version):
    X = diff_length_csv('2_5_day_nor_s2.csv')
    X = pad_sequences(X, maxlen=50, padding='post', truncating='post', value=0, dtype=float)
    X = np.array(X, dtype=float)
    nb_x_train = fun3(f_in='nb_x_train_%d.dat' % (data_version))
    nb_x_test = fun3(f_in='nb_x_test_%d.dat' % (data_version))
    y = fun2('number_category.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    lr = LogisticRegression(penalty='l1', fit_intercept=True, max_iter=200, warm_start=True)
    lr = lr.fit(X_train, y_train, )
    # print(lr.coef_)
    score = lr.score(X_test, y_test)
    print(score)
    return score

# score=0.0
# for data_version in range(1,6):
#     score += logistic_regression_prediction(data_version=data_version)
# print(score/5)

# split number_category.csv into two files according to its category
def split_category():
    f_in =open('number_category.csv')
    f_out_1=open('number_category_1.csv','w')
    f_out_2=open('number_category_2.csv','w')
    count =0
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        if line[1]=='1':
            count+=1
            f_out_1.write(lines)
        else:
            f_out_2.write(lines)

def re_split_train_test(f_in ='number_category_1.csv',mode='w'):
    f_in =open(f_in)
    number = []
    for i, lines in enumerate(f_in):
        line = lines.strip().split(',')[0]
        number.append(line)
    a = [i for i in range(0, len(number))]
    for k in range(1, 6):
        random.shuffle(a)
        f_out_train = open('nb_x_train_' + str(k) + '.dat', mode=mode)
        f_out_test = open('nb_x_test_' + str(k) + '.dat', mode=mode)
        for l in range(0, len(number)):
            if l < 0.7*len(number):
                f_out_train.write('%.1f\n' % float(number[a[l]]))
            else:
                f_out_test.write('%.1f\n' % float(number[a[l]]))

# re_split_train_test(f_in='number_category_1.csv',mode='w')
# re_split_train_test(f_in='number_category_2.csv',mode='a')


def get_next_temp(f_in ='2_5_day_nor_s2.csv'):
    f_in =open(f_in)
    f_out_1=open('front_1_10_temp.csv','w')
    f_out_2=open('front_2_11_temp.csv','w')
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        f_out_1.write(str(line[0]))
        f_out_2.write(str(line[0]))
        for k in range(1,11):
            try:
                f_out_1.write(','+str(line[k]))
            except IndexError:
                f_out_1.write(','+str(0))
        f_out_1.write('\n')
        for k in range(2,12):
            try:
                f_out_2.write(','+str(line[k]))
            except IndexError:
                f_out_2.write(','+str(0))
        f_out_2.write('\n')


# get_next_temp()