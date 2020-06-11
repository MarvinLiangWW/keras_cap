import numpy as np
from accessory import same_length_csv,read_case_nb,category_to_target
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import Activation
from keras.optimizers import Adam
from keras.regularizers import l2
from accessory import get_train_test_data
from matplotlib.pyplot import savefig,plot,legend,show,title,ylim
SEED=1337
np.random.seed(SEED)
batch_size = 16
nb_epochs = 200
nb_classes =2
time_steps =51
in_file_length =10
padding_value =0

def para_model():
    model = Sequential()
    model.add(Dense(16, input_dim=in_file_length, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(nb_classes,
                    W_regularizer=l2(0.01),
                    b_regularizer=l2(0.01)
                    ))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003),
                  metrics=['accuracy'])
    model.summary()

    return model

def para_prediction(model,data_version):
    X = same_length_csv('cap_feature_2.csv')
    y = same_length_csv('number_category.csv')
    nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat' % data_version)
    nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat' % data_version)
    x_train, x_test, y_train, y_test = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    probability_test = (sum(y_test) - len(y_test)) / len(y_test)
    print('probability_test:', probability_test)
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    model = model
    print('Train...')
    acc_list = []
    train_loss_list=[]
    test_loss_list=[]
    for epoch in range(nb_epochs):
        # print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1,validation_split=0.05 )
        train_score, train_acc = model.evaluate(x_train, y_train, batch_size=batch_size)
        score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        acc_list.append(acc)
        train_loss_list.append(train_score)
        test_loss_list.append(score)
        print('Test score:', score)
        print('Test accuracy:', acc)

    results = ""
    for i, acc in enumerate(acc_list):
        if acc > 0.72:
            if acc > 0.74:
                results += '\033[1;31m' + str(i + 1) + ':' + str(acc) + '\033[0m' + '; '
            else:
                results += '\033[1;34m' + str(i + 1) + ':' + str(acc) + '\033[0m' + '; '
        else:
            results += str(i + 1) + ':' + str(acc) + '; '
    plot(range(0, nb_epochs), acc_list,label='feature')
    plot(range(0, nb_epochs), train_loss_list,label='train_loss')
    plot(range(0, nb_epochs), test_loss_list,label='test_loss')
    acc_list = sorted(acc_list, reverse=True)
    print(acc_list)
    print("top-10 mean: %.3f" % np.mean(np.array(acc_list[:10])))
    print("top-50 mean: %.3f" % np.mean(np.array(acc_list[:50])))



if __name__ == '__main__':
    for data_version in range(1, 2):
        model = para_model()
        # model_study(model)
        para_prediction(model, data_version)
        legend()
        ylim((0.5, 1.0))
        show()
