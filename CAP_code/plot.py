import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_heatmap():

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

    f_pred =open('att_3_lstm_5_predicted_100.csv')
    pred =[]
    for i,lines in enumerate(f_pred):
        line =lines.strip().split(' ')[0]
        pred.append(line)
    print(pred)



    f_in=open('heatmap_2.dat')
    cate_1_1=[]
    cate_1_2=[]
    cate_2_1=[]
    cate_2_2=[]
    for i,lines in enumerate(f_in):
        line =lines.strip().split(' ')[1:]
        line =[float(l) for l in line]
        if category[i] == '1' and pred[i] =='1':
            line.append(0.15)
            cate_1_1.append(line)
        if category[i] =='1'and pred[i]=='2':
            line.append(0.01)
            cate_1_2.append(line)
        if category[i] =='2' and pred[i] =='1':
            line.append(0.01)
            cate_2_1.append(line)
        if category[i] =='2'and pred[i] =='2':
            line.append(0.15)
            cate_2_2.append(line)
    print(cate_2_2)
    print(np.array(cate_2_2).shape)
    data =np.concatenate([cate_1_1,cate_1_2,cate_2_2,cate_2_1],axis=0)

    print(np.array(data).shape)


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
    plt.show()
    # fig.savefig("nba.png")


def plot_heatmap_single(user):
    data=user.heatmap
    print(data)
    data= np.array(data).reshape((1,50))
    print(data.shape)
    sns.set(font_scale=1.0)
    # sns.set_style({"savefig.dpi": 100})
    # plot it out
    ax = sns.heatmap(data, cmap=plt.cm.Reds,)
    # set the x-axis labels on the top
    ax.xaxis.tick_top()
    # ax.set_yticklabels()
    # ax.yaxis.tick_top()
    ax.set_xticklabels(user.value,fontsize='small')
    for i in range(0,len(user.value)):
        if i>=50:
            break
        plt.text(i,1,user.value[i],size=7)

    plt.xlabel('temp position')
    plt.ylabel('patient '+str(user.id))
    # rotate the x-axis labels
    # plt.xticks(rotation=90)
    # get figure (usually obtained via "fig,ax=plt.subplots()" with matplotlib)
    fig = ax.get_figure()
    # specify dimensions and save
    fig.set_size_inches(15, 1)
    plt.show()
    # fig.savefig("nba.png")


def padding_list(padded_list,length):
    print(np.array(padded_list).shape)
    if length >= len(padded_list):
        padding =np.zeros(shape=(length - len(padded_list),))
        return np.concatenate((padded_list,padding),axis=0)
    else:
        return padded_list[0:length]

class user():
    def __init__(self,id):
        self.id=id
        self.temp =[]
        self.heatmap=[]
        self.value=[]


def plot_user(data):
    data.temp =padding_list(data.temp,50)
    # ax2 = plt.twinx()
    # plt.ylim([0,0.2])
    plt.plot(range(1,51),data.heatmap,label='heatmap')
    plt.scatter(range(1,51),data.temp,label='temp')
    plt.legend()
    plt.show()


def list_to_df(user):
    # rows=[]
    if len(user.value)>=50:
        heatmap = [[val] for val in user.heatmap[0:50]]
    else:
        heatmap = [[val] for val in user.heatmap]
    heatmap =np.array(heatmap).reshape((1,50))
    # for i in range(0,len(user.value)):
    #     if i>=50:
    #         break
    #     rows.append((heatmap[i]))
    # print(pd.DataFrame(rows))
    # print(pd.DataFrame(heatmap))
    return pd.DataFrame(heatmap)


def plot_heatmap_single_df(user,length):
    if len(user.value)>=length:
        heatmap = [[val] for val in user.heatmap[0:length]]
    else:
        heatmap = [[val] for val in user.heatmap]
    heatmap =np.array(heatmap).reshape((1,length))
    data =pd.DataFrame(heatmap)
    data.index.name = ""
    if len(user.value)>=length:
        labels=[val for val in user.value[0:length]]
    else:
        labels =[val for val in user.value]
    # print(labels)
    data.columns=labels

    sns.set(font_scale=1.0)
    # ax = sns.heatmap(data, cmap=plt.cm.Reds, linewidths=0,xticklabels=1,annot=np.array(user.value[0:length]).reshape((1,length)),fmt='.1f',robust=True)
    ax = sns.heatmap(data, cmap=plt.cm.Reds, linewidths=0,xticklabels=1,fmt='.1f',robust=True,square=True,cbar=False)
    ax.xaxis.tick_top()
    ax.set_yticks([])
    plt.xticks(rotation=90)
    # plt.yticks()

    fig = ax.get_figure()
    fig.set_size_inches(15, 1)
    plt.savefig('pic'+str(user.id)+'.pdf')
    # plt.show()


def age_distribution():
    f_num =open('Data/number_20180212.csv','r')
    num={}
    for i,lines in enumerate(f_num):
        line =lines.strip()
        num[line] = 1

    f_in =open('age.csv','r')

    mylist=[]
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')
        if line[0] in num.keys():
            mylist.append(float(line[1]))

    plt.hist(mylist,14)
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()


def padding():
    '''
    :return: 0.6482843137254902
    '''
    f_in =open('Data/temp_20180212.csv','r')
    length=0
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')
        if len(line)>=25:
            length+=24
        else:
            length+=len(line)-1
    print(length/i/24)


def categroty():
    label =[]
    f_label = open('category_3.csv','r')
    for i,lines in enumerate(f_label):
        line =lines.strip()
        label.append(int(line))
    print(label)
    num =[]
    f_num =open('Number_1277.csv','r')
    for i,lines in enumerate(f_num):
        line =lines.strip()
        num.append(int(line))

    dict={}
    count =np.zeros(shape=(3,))
    f_num = open('Data/number_20180212.csv','r')
    for i,lines in enumerate(f_num):
        line =lines.strip()
        if int(line) in num:
            _index = num.index(int(line))
            dict[line] = int(label[_index])
            count[int(label[_index])-1]+=1
    print(count)
    print(count/sum(count))



def final_cate():
    f_in =open('Data/label_20180212.csv','r')
    count=0
    for i,lines in enumerate(f_in):
        line=lines.strip().split(',')
        count+=int(line[1])
    print((count-681)/681)

if __name__=='__main__':
    print('main')
    # padding()
    # age_distribution()
    # final_cate()
    categroty()





    # get_temp('temp_raw.csv','number_category.csv','label_20180212.csv','temp_20180212.csv')
    # plot__('temperature_raw.csv','number_category.csv')




    #
    # data_28347 =user(id=28347)
    # data_28347.temp =[0.286,0.393,0.357,1.000,0.214,0.000,1.000,1.000,0.000,1.000,1.000,0.000,1.000,1.000,0.000,0.143,0.036,1.000,0.321,0.000,0.143,1.000,0.250,1.000,1.000,0.429,0.000,0.036,0.357,0.000,0.107,0.071,0.429,0.000,0.000,0.107,0.000,0.000,0.000,0.000,0.000]
    # data_28347.heatmap=[0.01580, 0.01696, 0.01473, 0.00696, 0.00769, 0.00522,
    #                     0.00454, 0.00393, 0.00350, 0.00482, 0.00312, 0.00291,
    #                     0.00293, 0.00478, 0.00351, 0.00283, 0.00298, 0.00432,
    #                     0.00392, 0.00308, 0.00430, 0.00326, 0.00412, 0.00327,
    #                     0.00300, 0.00662, 0.00281, 0.00321, 0.00404, 0.00368,
    #                     0.00345, 0.00559, 0.00564, 0.00386, 0.00513, 0.00742,
    #                     0.00648, 0.00481, 0.00752, 0.00614, 0.00597, 0.00645,
    #                     0.00874, 0.00758, 0.00928, 0.00909, 0.01006, 0.01178,
    #                     0.01590, 0.01469, 0.04638, 0.02054, 0.03799, 0.03228,
    #                     0.03806, 0.10315, 0.08152, 0.06826, 0.10065, 0.15879]
    #
    # plot_user(data_28347)
    #
