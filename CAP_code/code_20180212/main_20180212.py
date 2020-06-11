# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from keras.preprocessing.sequence import pad_sequences
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from keras.models import Sequential
import argparse

from code_20180212.model_20180212 import *

# from model_20180212 import *

batch_size = 16
nb_epochs = 200


def plot__(filename, filelabel):
    f_in = open(filename, 'r')
    # for i,lines in enumerate(f_in):
    #     line =lines.strip().split(',')
    #     plt.title(str(line[0]))
    #     plt.plot(range(0,len(line)-1),line[1:])
    #     plt.show()

    label = []
    with open(filelabel, 'r') as f_label:
        for i, lines in enumerate(f_label):
            line = lines.strip().split(',')
            label.append(line[1])
    for i, lines in enumerate(f_in):
        line = lines.strip().split(',')
        # plt.title(str(line[0]))
        plt.subplot(10, 10, i % (10 * 10) + 1)
        plt.ylim(37.2, 40.0)
        if label[i] == '1':
            plt.plot(range(0, len(line) - 1), line[1:], color='r')
        else:
            plt.plot(range(0, len(line) - 1), line[1:], color='b')
        if (i + 1) % 100 == 0:
            plt.show()


def get_temp(filename, file_label, file_outLabel, file_outTemp):
    label = []
    with open(file_label, 'r') as f_label:
        for i, lines in enumerate(f_label):
            line = lines.strip().split(',')
            label.append(line)

    temp = {}
    with open(filename, 'r') as f_input:
        for i, lines in enumerate(f_input):
            line = lines.strip().split(',')
            temp[line[0]] = line[1:]
    print('temp.shape', len(temp))

    new_label = []
    new_temp = {}
    for index in range(0, len(label)):
        try:
            new_temp[label[index][0]] = temp[label[index][0]]
            new_label.append(label[index])
        except KeyError:
            pass
    print('new_temp.shape', len(new_temp))

    f_outTemp = open(file_outTemp, 'w', encoding='utf-8')
    f_outLabel = open(file_outLabel, 'w', encoding='utf-8')
    for index in range(0, len(new_label)):
        f_outLabel.write('%s,%s\n' % (new_label[index][0], new_label[index][1]))
        f_outTemp.write('%s' % (new_label[index][0]))
        for item in new_temp[new_label[index][0]]:
            f_outTemp.write(',%s' % (item))
        f_outTemp.write('\n')
    f_outTemp.close()
    f_outLabel.close()


def temp_stastics(filename):
    dict = {}
    with open(filename, 'r', encoding='utf-8') as f_in:
        for i, lines in enumerate(f_in):
            line = lines.strip().split(',')
            dict[line[0]] = line[1:]

    len_list = np.zeros(shape=(50,))
    max_len = 0
    avg_len = 0
    for key in dict.keys():
        avg_len += len(dict[key])
        len_list[len(dict[key])] += 1
        if len(dict[key]) > max_len:
            max_len = len(dict[key])
    print('max_len:', max_len)
    print('avg_len:', avg_len / len(dict.keys()))

    plt.subplot(2, 1, 1)
    plt.plot(range(0, len(len_list)), len_list)
    plt.xlabel('temp length')
    plt.ylabel('count')
    plt.title('max:%d avg:%.5f' % (max_len, avg_len / len(dict.keys())))

    plt.subplot(2, 1, 2)
    plt.plot(range(0, len(len_list)), [sum(len_list[0:i]) / sum(len_list) for i in range(0, len(len_list))])
    plt.xlabel('temp_length')
    plt.ylabel('percentage')
    plt.show()


def get_feature(filename, file_label, file_outFeature):
    label = []
    with open(file_label, 'r') as f_label:
        for i, lines in enumerate(f_label):
            line = lines.strip().split(',')
            label.append(line)

    feature = {}
    with open(filename, 'r', encoding='utf-8') as f_feature:
        for i, lines in enumerate(f_feature):
            line = lines.strip().split(',')
            feature[str(int(float(line[0])))] = line[1:]

    f_outFeature = open(file_outFeature, 'w', encoding='utf-8')
    for i in range(0, len(label)):
        try:
            feature_list = feature[label[i][0]]
            f_outFeature.write(label[i][0])
            for k in range(0, len(feature_list)):
                f_outFeature.write(',%s' % feature_list[k])
            f_outFeature.write('\n')
        except KeyError:
            pass
    f_outFeature.close()


def get_number(filename, f_output):
    f_out = open(f_output, 'w', encoding='utf-8')
    with open(filename, 'r') as f_in:
        for i, lines in enumerate(f_in):
            line = lines.strip().split(',')
            f_out.write(line[0] + '\n')
    f_out.close()


def nb_5_cv_split(f_in):
    f_in = open(f_in)
    number = []
    for i, lines in enumerate(f_in):
        line = lines.strip()
        number.append(line)
    number = np.array(number)
    np.random.shuffle(number)

    for i in range(1, 6):
        f_train_cv1 = open('../Data/nb_train_cv%d.dat' % (i), 'w')
        f_test_cv1 = open('../Data/nb_test_cv%d.dat' % (i), 'w')
        for j in range(0, number.shape[0]):
            t = j % 5
            if t != i - 1:
                f_train_cv1.write(number[j] + '\n')
            else:
                f_test_cv1.write(number[j] + '\n')
        f_train_cv1.close()
        f_test_cv1.close()


def diff_length_csv(filename):
    f_in = open(filename)
    return_list = []
    for i, lines in enumerate(f_in):
        line = lines.strip().split(',')[0:]
        return_list.append(line)
    return return_list


def read_csv2list(filename):
    f_in = open(filename)
    return_list = []
    for i, lines in enumerate(f_in):
        line = lines.strip().split(',')[1:]
        return_list.append(line)
    return return_list


def read_csv2list2(filename):
    f_in = open(filename)
    return_list = []
    for i, lines in enumerate(f_in):
        line = lines.strip().split(',')[1:]
        line = np.asarray(line, dtype=np.float32)
        return_list.append(list(line))
    return return_list


def get_train_test_data(X_con, y, nb_x_train, nb_x_test, ):
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
    return np.array(X_train, ), np.array(X_test), np.array(y_train), np.array(y_test),


def read_csv2array(filename):
    result = []
    with open(filename, 'r', encoding='utf-8') as f_in:
        for i, lines in enumerate(f_in):
            line = lines.strip().split(',')
            result.append(line)
    return np.array(result, dtype=np.float32)


def read_index(filename, split=','):
    result = []
    with open(filename, 'r', encoding='utf-8') as f_in:
        for i, lines in enumerate(f_in):
            line = lines.strip().split(split)
            result.append(line[0])
    return np.array(result, dtype=np.int32)


def lr_prediction(data_version, Temp, Feature, time_steps, penalty='l2', tol=1e-4):
    if Temp:
        temp = read_csv2list('../Data/temp_20180212.csv')
        pad_temp = np.array(
            pad_sequences(temp, maxlen=time_steps, padding='post', truncating='post', value=0, dtype=np.float32),
            dtype=np.float32)
    else:
        pad_temp = None

    if Feature:
        feature = read_csv2array('../Data/cap_feature_20180212.csv')
    else:
        result = read_index('../Data/cap_feature_20180212.csv')
        feature = np.reshape(result, newshape=(result.shape[0], 1))
    if pad_temp is None:
        X = feature
    else:
        X = np.concatenate((feature, pad_temp), axis=1)
    print('input shape:', X.shape)
    nb_x_train = read_index('../Data/nb_train_cv%d.dat' % (data_version))
    nb_x_test = read_index('../Data/nb_test_cv%d.dat' % (data_version))
    y = read_csv2array('../Data/label_20180212.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    lr = LogisticRegression(penalty=penalty, fit_intercept=True, max_iter=200, warm_start=True, tol=tol)
    lr = lr.fit(X_train, y_train, )
    score = lr.score(X_test, y_test)
    y_pre = lr.predict(X_test)
    roc_auc = metrics.roc_auc_score(y_test - 1, y_pre - 1)
    print('acc: ', score, ' auc: ', roc_auc)
    return score, roc_auc


def gbdt_prediction(data_version, Temp, Feature, time_steps, lr=1e-4):
    if Temp:
        temp = read_csv2list('../Data/temp_20180212.csv')
        pad_temp = np.array(
            pad_sequences(temp, maxlen=time_steps, padding='post', truncating='post', value=0, dtype=np.float32),
            dtype=np.float32)
    else:
        pad_temp = None

    if Feature:
        feature = read_csv2array('../Data/cap_feature_20180212.csv')
    else:
        result = read_index('../Data/cap_feature_20180212.csv')
        feature = np.reshape(result, newshape=(result.shape[0], 1))
    if pad_temp is None:
        X = feature
    else:
        X = np.concatenate((feature, pad_temp), axis=1)

    nb_x_train = read_index('../Data/nb_train_cv%d.dat' % (data_version))
    nb_x_test = read_index('../Data/nb_test_cv%d.dat' % (data_version))
    y = read_csv2array('../Data/label_20180212.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    print('X-train shape:', X_train.shape)
    lr = GradientBoostingClassifier(n_estimators=100, learning_rate=lr, max_depth=3)
    lr = lr.fit(X_train, y_train, )
    score = lr.score(X_test, y_test)
    y_pre = lr.predict(X_test)
    roc_auc = metrics.roc_auc_score(y_test - 1, y_pre - 1)
    print('dataset: ', data_version, 'acc: ', score, 'auc: ', roc_auc)
    return score, roc_auc


def lr_avgTemp_prediction(data_version, Temp, Feature, penalty='l2', tol=1e-4):
    if Temp:
        temp = read_csv2list2('../Data/temp_20180212.csv')
        pad_temp = np.zeros(shape=(len(temp), 2), )
        for k in range(0, len(temp)):
            pad_temp[k][0] = sum(temp[k]) / len(temp[k])
            pad_temp[k][1] = max(temp[k])
            # pad_temp[k][2] =min(temp[k])
    else:
        pad_temp = None

    if Feature:
        feature = read_csv2array('../Data/cap_feature_20180212.csv')
    else:
        result = read_index('../Data/cap_feature_20180212.csv')
        feature = np.reshape(result, newshape=(result.shape[0], 1))
    if pad_temp is None:
        X = feature
    else:
        X = np.concatenate((feature, pad_temp), axis=1)
    print('input shape:', X.shape)
    nb_x_train = read_index('../Data/nb_train_cv%d.dat' % (data_version))
    nb_x_test = read_index('../Data/nb_test_cv%d.dat' % (data_version))
    y = read_csv2array('../Data/label_20180212.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    lr = LogisticRegression(penalty=penalty, fit_intercept=True, max_iter=200, warm_start=True, tol=tol)
    lr = lr.fit(X_train, y_train, )
    score = lr.score(X_test, y_test)
    y_pre = lr.predict(X_test)
    roc_auc = metrics.roc_auc_score(y_test - 1, y_pre - 1)
    print('acc: ', score, ' auc: ', roc_auc)
    return score, roc_auc


def gbdt_avgTemp_prediction(data_version, Temp, Feature, lr=1e-4):
    if Temp:
        temp = read_csv2list2('../Data/temp_20180212.csv')
        pad_temp = np.zeros(shape=(len(temp), 1), )
        for k in range(0, len(temp)):
            pad_temp[k][0] = sum(temp[k]) / len(temp[k])
            # pad_temp[k][1]= max(temp[k])
            # pad_temp[k][2] =min(temp[k])
    else:
        pad_temp = None

    if Feature:
        feature = read_csv2array('../Data/cap_feature_20180212.csv')
    else:
        result = read_index('../Data/cap_feature_20180212.csv')
        feature = np.reshape(result, newshape=(result.shape[0], 1))
    if pad_temp is None:
        X = feature
    else:
        X = np.concatenate((feature, pad_temp), axis=1)
    print('input shape:', X.shape)
    nb_x_train = read_index('../Data/nb_train_cv%d.dat' % (data_version))
    nb_x_test = read_index('../Data/nb_test_cv%d.dat' % (data_version))
    y = read_csv2array('../Data/label_20180212.csv')
    X_train, X_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    lr = GradientBoostingClassifier(n_estimators=100, learning_rate=lr, max_depth=3)
    lr = lr.fit(X_train, y_train, )
    score = lr.score(X_test, y_test)
    y_pre = lr.predict(X_test)
    roc_auc = metrics.roc_auc_score(y_test - 1, y_pre - 1)
    return score, roc_auc


def print_score(time_steps):
    score = 0.0
    auc = 0.0
    for data_version in range(1, 6):
        # rt_acc, rt_auc = lr_prediction(data_version=data_version,Temp=True,Feature=True,penalty='l2',time_steps=time_steps,tol=0.1)
        # rt_acc, rt_auc = lr_prediction(data_version=data_version,Temp=True,Feature=False,time_steps=time_steps,penalty='l2',tol=0.01)
        # rt_acc, rt_auc = lr_prediction(data_version=data_version,Temp=False,Feature=True,penalty='l2',tol=0.001)

        # rt_acc, rt_auc = gbdt_prediction(data_version=data_version,Temp=True,Feature=True,time_steps=time_steps,lr=0.01)
        # rt_acc, rt_auc = gbdt_prediction(data_version=data_version,Temp=True,Feature=False,time_steps=time_steps,lr=0.0001)
        # rt_acc, rt_auc = gbdt_prediction(data_version=data_version,Temp=False,Feature=True,lr=0.01)

        rt_acc, rt_auc = lr_avgTemp_prediction(data_version, Temp=True, Feature=True, penalty='l2', tol=0.25)
        # rt_acc, rt_auc = gbdt_avgTemp_prediction(data_version, Temp=True, Feature=True,lr=0.01)


        score += rt_acc
        auc += rt_auc
    print('acc', score / 5)
    print('auc', auc / 5)
    return score / 5, auc / 5


def category_to_target(category):
    y = []
    for i in range(0, category.shape[0]):
        temp = []
        for k in range(0, nb_classes):
            if k + 1 == category[i]:
                temp.append(1)
            else:
                temp.append(0)
        y.append(temp)
    return np.array(y, dtype=type(y[0][0]))


def feature_prediction(data_version):
    X = read_csv2array('../Data/cap_feature_20180212.csv')
    y = read_csv2array('../Data/label_20180212.csv')
    nb_x_train = read_index('../Data/nb_train_cv%d.dat' % (data_version))
    nb_x_test = read_index('../Data/nb_test_cv%d.dat' % (data_version))

    x_train, x_test, y_train, y_test = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    probability_test = (sum(y_test) - len(y_test)) / len(y_test)

    print('probability_test:', probability_test)
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    model = Sequential()
    model.add(Dense(8, input_dim=feature_length, activation='relu'))
    # model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(nb_classes, kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01),
                    ))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003),
                  metrics=['accuracy'])
    model.summary()

    print('Train...')
    acc_list = []
    train_loss_list = []
    test_loss_list = []
    auc_list = []
    for epoch in range(nb_epochs):
        # print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1)
        train_score, train_acc = model.evaluate(x_train, y_train, batch_size=batch_size)
        score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        acc_list.append(acc)
        train_loss_list.append(train_score)
        test_loss_list.append(score)
        print('Test score:', score)
        print('Test accuracy:', acc)
        y_pre = model.predict(x_test, )
        roc_auc = metrics.roc_auc_score(y_test, y_pre)
        print(roc_auc)
        auc_list.append(roc_auc)

    plt.plot(range(0, nb_epochs), acc_list, label='feature')
    plt.plot(range(0, nb_epochs), train_loss_list, label='train_loss')
    plt.plot(range(0, nb_epochs), test_loss_list, label='test_loss')

    acc_list = sorted(acc_list, reverse=True)
    print(acc_list)
    plt.title('temp_study_%d' % (data_version))
    print('temp_study_%d\n' % (data_version))
    print("top-10 mean: %.5f" % np.mean(np.array(acc_list[:10])))
    print("top-50 mean: %.5f" % np.mean(np.array(acc_list[:50])))

    auc_list_sorted = sorted(auc_list, reverse=True)
    print("top-10 mean: %.5f" % np.mean(np.array(auc_list_sorted[:10])))
    print("top-50 mean: %.5f" % np.mean(np.array(auc_list_sorted[:50])))

    return np.mean(np.array(acc_list[:50])), np.mean(np.array(auc_list_sorted[:50]))


def model_study(model, data_version, time_steps):
    X = diff_length_csv('../Data/temp_20180212.csv')
    X = pad_sequences(X, maxlen=time_steps, padding='post', truncating='post', value=PADDING_VALUE, dtype=float)
    print(X.shape)
    y = read_csv2array('../Data/label_20180212.csv')
    nb_x_train = read_index('../Data/nb_train_cv%d.dat' % (data_version))
    nb_x_test = read_index('../Data/nb_test_cv%d.dat' % (data_version))

    x_train, x_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    print('class 2', (sum(y_test) - len(y_test)) / len(y_test))

    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    model = model

    # test
    acc_list = []
    train_score_list = []
    test_score_list = []
    auc_list = []
    for epoch in range(nb_epochs):
        print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1, shuffle=True, verbose=1)
        train_score, train_acc = model.evaluate(x_train, y_train, batch_size=batch_size)
        score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        y_pre = model.predict(x_test, )
        roc_auc = metrics.roc_auc_score(y_test, y_pre)
        print(roc_auc)
        auc_list.append(roc_auc)
        acc_list.append(acc)
        train_score_list.append(train_score)
        test_score_list.append(score)
        print('Test score:', score)
        print('Test accuracy:', acc)
    plt.plot(range(0, nb_epochs), acc_list, label='temp_acc')
    plt.plot(range(0, nb_epochs), train_score_list, label='train_loss')
    plt.plot(range(0, nb_epochs), test_score_list, label='test_loss')
    plt.plot(range(0, nb_epochs), auc_list, label='auc')

    acc_list = sorted(acc_list, reverse=True)
    plt.title('temp_study_%d' % (data_version))
    print('temp_study_%d\n' % (data_version))
    print("top-10 acc mean: %.5f" % np.mean(np.array(acc_list[:10])))
    print("top-50 acc mean: %.5f" % np.mean(np.array(acc_list[:50])))

    # plt.savefig('temp_study_%d'%(data_version))
    auc_list_sorted = sorted(auc_list, reverse=True)
    print("top-10 auc mean: %.5f" % np.mean(np.array(auc_list_sorted[:10])))
    print("top-50 auc mean: %.5f" % np.mean(np.array(auc_list_sorted[:50])))
    plt.legend()
    plt.show()
    return np.mean(np.array(acc_list[:50])), np.mean(np.array(auc_list_sorted[:50]))


def merge_model_study(model, data_version, time_steps):
    X = diff_length_csv('../Data/temp_20180212.csv')
    X = pad_sequences(X, maxlen=time_steps, padding='post', truncating='post', value=PADDING_VALUE, dtype=float)
    print(X.shape)
    y = read_csv2array('../Data/label_20180212.csv')
    nb_x_train = read_index('../Data/nb_train_cv%d.dat' % (data_version))
    nb_x_test = read_index('../Data/nb_test_cv%d.dat' % (data_version))
    x_train, x_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)
    # use y_test_ for calculate auc

    X2 = read_csv2array('../Data/cap_feature_20180212.csv')
    x2_train, x2_test, y2_train, y2_test, = get_train_test_data(X2, y, nb_x_train, nb_x_test, )

    model = model

    # test
    acc_list = []
    train_score_list = []
    test_score_list = []
    auc_list = []
    for epoch in range(nb_epochs):
        '''
        if epoch ==100:
            get_3rd_layer_output = K.function([model.get_layer('input_temp').input,model.get_layer('input_para').input,K.learning_phase()],[model.get_layer(name='AttLayer').heatmap])
            output =get_3rd_layer_output([x_test,x2_test,0])
            print(type(output))
            print(output)
            print(output[0].shape)
            f_out =open('heatmap_2.dat','w')
            for i in range(0,len(output[0][0])):
                f_out.write('%d' % (i))
                for m in range(0,len(output[0])):
                    f_out.write(' %.5f'%(output[0][m][i]))
                f_out.write('\n')
            for i in range(0,len(output[0])):
                f_out.write('%d' % (i))
                for m in range(0,len(output[0][0])):
                    f_out.write(' %.5f'%(output[0][i][m]))
                f_out.write('\n')
            predicted= model.predict([x_test, x2_test], batch_size=batch_size)
            print(predicted.shape)
            f_out=open('att_3_lstm_5_predicted_100.csv','w')
            for i in range(0,len(predicted)):
                if predicted[i][0]>0.5:
                    f_out.write('1\n')
                else:
                    f_out.write('2\n')
            f_out.close()
        '''

        print('Train...')
        model.fit([x_train, x2_train], y_train, batch_size=batch_size, epochs=1, shuffle=True, verbose=1)
        train_score, train_acc = model.evaluate([x_train, x2_train], y_train, batch_size=batch_size, )

        score, acc = model.evaluate([x_test, x2_test], y_test, batch_size=batch_size)
        y_pre = model.predict([x_test, x2_test], )
        roc_auc = metrics.roc_auc_score(y_test, y_pre)

        acc_list.append(acc)
        train_score_list.append(train_score)
        test_score_list.append(score)
        auc_list.append(roc_auc)
        print('Test score:', score)
        print('Test accuracy:', acc)
    plt.plot(range(0, nb_epochs), acc_list, label='temp_acc')
    plt.plot(range(0, nb_epochs), train_score_list, label='train_loss')
    plt.plot(range(0, nb_epochs), test_score_list, label='test_loss')
    plt.plot(range(0, nb_epochs), auc_list, label='auc')

    acc_list = sorted(acc_list, reverse=True)
    plt.title('temp_study_%d' % (data_version))
    print('temp_study_%d\n' % (data_version))
    print("top-10 acc mean: %.3f" % np.mean(np.array(acc_list[:10])))
    print("top-50 acc mean: %.3f" % np.mean(np.array(acc_list[:50])))

    # plt.savefig('temp_study_%d'%(data_version))
    auc_list_sorted = sorted(auc_list, reverse=True)
    print("top-10 auc mean: %.3f" % np.mean(np.array(auc_list_sorted[:10])))
    print("top-50 auc mean: %.3f" % np.mean(np.array(auc_list_sorted[:50])))
    plt.legend()
    plt.show()
    return np.mean(np.array(acc_list[:50])), np.mean(np.array(auc_list_sorted[:50]))


def parse_args():
    parser = argparse.ArgumentParser(description="Run Deep Learning for Rating.")
    parser.add_argument('--model', nargs='?', default='merge_att_3_lstm_5',
                        help="e.g. 'lstm' ")
    parser.add_argument('--dataset', type=int, default=1,
                        help='e.g. 1 Choose from [1,2,3,4,5]')
    parser.add_argument('--type', nargs='?', default='merge',
                        help="e.g. 'feature' Choose from ['feature','temp','merge']")
    parser.add_argument('--time_steps', type=int, default=51,
                        help="e.g. 50 Choose from [11-51]")
    parser.add_argument('--bus', nargs='?', default='0',
                        help="e.g, '0' Determine to use which GPU")
    return parser.parse_args()


if __name__ == '__main__':
    print('main')
    # get_feature('cap_feature.csv','label_20180212.csv','cap_feature_20180212.csv')
    # get_temp('temp_raw.csv','number_category.csv','label_20180212.csv','temp_20180212.csv')
    # plot__('temperature_raw.csv','number_category.csv')
    # temp_stastics('temp_20180212.csv')
    # get_number('../code_20180212/label_20180212.csv','../code_20180212/number_20180212.csv')
    # nb_5_cv_split('../Data/number_20180212.csv',)

    # count = 0.0
    # auc = 0.0
    # time_steps = 24
    # for i in range(1, 2):
    #     acc, auc_ = print_score(time_steps=time_steps)
    #     count += acc
    #     auc += auc_
    # print('acc', count / 10)
    # print('auc', auc / 10)
    # print(time_steps)
    # exit()


    '''
    for lr and gbdt
    '''
    # acc_list=[]
    # auc_list=[]
    # for k in range(10,51):
    #     count = 0.0
    #     auc = 0.0
    #     time_steps = k
    #     for i in range(1, 2):
    #         acc, auc_ = print_score(time_steps=time_steps)
    #         count += acc
    #         auc += auc_
    #     print('acc', count / 10)
    #     print('auc', auc / 10)
    #     print(time_steps)
    #     acc_list.append(count/10)
    #     auc_list.append(auc/10)
    # plt.plot(range(10,51),acc_list,label='acc')
    # plt.plot(range(10,51),auc_list,label='auc')
    # plt.show()
    # exit()


    args = parse_args()
    TYPE = args.type
    MODEL = args.model
    DATA = args.dataset
    TIME_STEPS = args.time_steps

    bus = args.bus
    if bus == None:
        pass
    else:
        import os

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = bus
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.Session(config=config)

    avg_acc = 0
    avg_auc = 0

    if TYPE == 'feature':
        avg_acc, avg_auc = feature_prediction(DATA)
    elif TYPE == 'temp':
        if MODEL == 'model_lstm':
            model = model_lstm(TIME_STEPS)
        else:
            model = model_lstm(TIME_STEPS)
        avg_acc, avg_auc = model_study(model, DATA, TIME_STEPS)
    elif TYPE == 'merge':
        if MODEL == 'merge_lstm':
            model = merge_lstm(TIME_STEPS)
        elif MODEL == 'merge_att_3_lstm_5':
            model = merge_att_3_lstm_5(TIME_STEPS)
        else:
            model = merge_lstm(TIME_STEPS)
        avg_acc, avg_auc = merge_model_study(model, DATA, TIME_STEPS)

    # print hyper-parameter
    print('*********************')
    print('model: ', MODEL)
    print('type: ', TYPE)
    print('dataset: ', DATA)
    print('time_steps:', TIME_STEPS)
    print('average acc: ', '%.5f' % avg_acc)
    print('average auc: ', '%.5f' % avg_auc)
    print('*********************')
