import csv
from datetime import datetime
from sklearn.cluster import KMeans
import random
import numpy as np
from sklearn import metrics
np.random.seed(40)
def getnumber_category():
    csv_reader1 = csv.reader(open('2015-10-5huizong.csv'))
    '''GBK的原因：支（衣）原体肺炎在数据中有‘1’和‘衣原体’两种标识'''
    f = open('number_category.csv', 'w+')
    for i in csv_reader1:
        x1 = i[0]
        x4 = i[4]
        x5 = i[5]
        x6 = i[6]
        x7 = i[7]
        x8 = i[8]

        if x4 != '' and x5 == '' and x6 == '':
            f.write(x1 + ',1' + '\n')
            continue
        if x5 != '' and x4 == '' and x6 == '':
            f.write(x1 + ',2' + '\n')
            continue
        if x6 != '' and x4 == '' and x5 == '':
            f.write(x1 + ',3' + '\n')
            continue
    '''有不是肺炎的情况'''
    '''有数据只有4未给出具体是哪几种组合的情况'''
    '''有数据给出1和2组合未标注其为4的情况'''
    f.close()


def splittwocol(filename,outputcol1,ouputcol2,mode):
    csv_reader = csv.reader(open(filename))
    f1=open(outputcol1,mode)
    f2 = open(ouputcol2,mode)
    for i in csv_reader:
        f1.write(i[0]+'\n')
        f2.write(i[1]+'\n')
    f1.close()
    f2.close()

def get_number_age(filename,output,mode):
    csv_reader = csv.reader(open(filename))
    f = open(output, mode)
    for i in csv_reader:
        birth_date = datetime.strptime(i[1], "%Y/%m/%d")
        start_date = datetime.strptime(i[2], "%Y/%m/%d %H:%M")
        time = (start_date - birth_date).days
        f.write(i[0] + ','),
        temp = '%.2f' % (time / 365)
        f.write(str(temp) + '\n')
    f.close()

def merge_tiwen_number_fun(f_in='5_day_temp.csv',f_in2='Number.csv',f_out='5_day_tiwencheck.csv',mode='w'):
    f = open(f_in)
    csv_reader2 = csv.reader(open(f_in2))
    number2=[k[0] for k in csv_reader2]
    f_out = open(f_out,mode)
    for i, line in enumerate(f):
        all =line.strip().split(' ')
        number=all[0]
        for k in range(0,len(number2)):
            if number == number2[k]:
                f_out.write(str(number))
                for j in range(1,len(all)):
                    f_out.write(',')
                    f_out.write(str(all[j]))
                f_out.write('\n')
                break
    f.close()
    f_out.close()






def add_tiwen_fun(filename1,filename2,output,mode):
    csv_reader = csv.reader(open(filename1))
    csv_reader2 = csv.reader(open(filename2))
    number=[]
    parameter=[]
    for i in csv_reader:
        number.append(i[0])
        k=1
        while k<len(i):
            parameter.append(i[k])
            k+=1
    f = open(output,mode)
    for i in csv_reader2:
        x = i[0]
        f.write(x)
        b=1
        while b<len(i):
            f.write(','+i[b])
            b+=1

        lennumber = len(number)
        a=0
        while a<lennumber:
            if number[a] == x:
                t=0
                while t<12:
                    f.write(','+parameter[a*12+t])
                    t+=1
                break
            a+=1
        if a == lennumber:
            f.write(','+ '0')
        f.write('\n')
    f.close()


#dimension1 commonly choose 3
def add_data_fun(filename1,dimension1,filename2,output,mode):
    csv_reader = csv.reader(open(filename1))
    csv_reader2 = csv.reader(open(filename2))
    number=[]
    parameter=[]
    for i in csv_reader:
        number.append(i[0])
        k = 1
        while k < len(i):
            parameter.append(i[k])
            k += 1
    f = open(output,mode)
    for i in csv_reader2:
        x = i[0]
        f.write(x)
        b=1
        while b<len(i):
            f.write(','+i[b])
            b+=1

        lennumber = len(number)
        a=0
        while a<lennumber:
            if number[a] == x:
                t = 0
                while t < dimension1:
                    f.write(',' + parameter[a * dimension1 + t])
                    t += 1
                break
            a+=1
        if a == lennumber:
            t = 0
            while t < dimension1:
                f.write(',' + '-1')
                t += 1
        f.write('\n')
    f.close()


def fun(filename,testname,output,mode):
    csv_reader = csv.reader(open(filename))
    f = open(output, mode)
    for i in csv_reader:
        if i[3] == testname and testname =='白蛋白':
            try:
                if float(i[4])<1.00:
                    f.write('')
                elif 54.00>=float(i[4])>=38.00:
                    f.write(i[0]+','+i[4]+',1\n')
                else:
                    f.write(i[0] + ',' + i[4] + ',0\n')
            except ValueError:
                f.write('')
            except IndexError:
                f.write('')
        elif i[3] == testname and testname =='红细胞压积':
            try:
                if 0.20<=float(i[4])<=1.00:
                    i[4] = str(100 * float(i[4]))
                if 48.00>=float(i[4])>=34.00:
                    f.write(i[0]+','+i[4]+',1\n')
                else:
                    f.write(i[0] + ',' + i[4] + ',0\n')
            except ValueError:
                f.write('')
        elif i[3] == testname and testname =='红细胞分布宽度变异系数':
            try:
                if 0.01<=float(i[4])<=0.99:
                    i[4]=str(100*float(i[4]))
                if 16.00>=float(i[4])>=11.00:
                    f.write(i[0]+','+i[4]+',1\n')
                else:
                    f.write(i[0] + ',' + i[4] + ',0\n')
            except ValueError:
                f.write('')
        elif i[3] ==testname:
            # f.write(i[0])
            s=i[5].split('～')
            s0 ='%.2f'%(float(s[0]))
            s1 ='%.2f'%(float(s[1]))
            try:
                if float(s1)>=float(i[4])>=float(s0):
                    f.write(i[0]+','+i[4]+',1\n')
                else:
                    f.write(i[0] + ',' + i[4] + ',0\n')
            except ValueError:
                f.write('')
    f.close()


def average_value(filename):
    csv_reader = csv.reader(open(filename))
    value = [i[1] for i in csv_reader]
    count_value =0
    for k in range (0,len(value)):
        count_value+=float(value[k])
    return '%.3f'%(count_value/len(value))


def add0fun(filename,filename2,col,output,mode):
    csv_reader = csv.reader(open(filename))
    number = []
    parameter = []
    for i in csv_reader:
        number.append(i[0])
        for k in range(1,len(i)):
            parameter.append(i[k])
    f = open(output, mode)
    csv_reader2 = csv.reader(open(filename2))
    number2 = [i[0] for i in csv_reader2]
    for k in range(0,len(number2)):
        f.write(number2[k])
        j=0
        while j < len(number):
            if number2[k]==number[j]:
                for l in range(0,col):
                    f.write(','+parameter[j*col+l])
                break
            j+=1
        if j==len(number):
            f.write(','+average_value(filename))
            f.write(','+'1')
            # pretend that the average value of test is always normal
        f.write('\n')


def invoke_fun(dataset,testname,output,mode,number,outputtran,lastoutput,finaloutput):
    fun(dataset, testname, output, mode)
    add0fun(output, number, 2, outputtran, mode)
    add_data_fun(outputtran, 2, lastoutput, finaloutput, mode)

def get_csvdata(filename1,filename2):
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

def kmeans(n_clusters):
    X, y = get_csvdata('tiwencheck.csv', 'Category.csv')
    # reduced_data = PCA(n_components=2).fit_transform(X)
    km = KMeans(n_clusters=n_clusters).fit(X)
    y_pred = km.predict(X)
    print(metrics.accuracy_score(y,y_pred))
    csv_reader = csv.reader(open('Number.csv'))
    number = [i[0] for i in csv_reader]
    f = open('tiwenlabel.csv','w')
    for i in range(0,len(y_pred)):
        f.write(number[i])
        for k in range(0,n_clusters):
            if k==y_pred[i]:
                f.write(',1')
            else:
                f.write(',0')
        f.write('\n')

def xman_xmix_normalization(f_in='number_age_col55.csv',f_out='number_age_col55tran.csv',rows=1277,cols=58):
    f_in = open(f_in)
    R = np.mat(np.zeros((rows,cols)))
    for i, line in enumerate(f_in):
        segs = line.strip().split(',')[0:]
        for k in range(0,len(segs)):
            R[i, k] = float(segs[k])
    for t in range(1,cols):
        col_t = R[:,t]
        x_max = float(col_t.max(axis=0))
        x_min = float(col_t.min(axis=0))
        if x_max-x_min!=0:
            x_length=x_max-x_min
        for row in range(0,rows):
            R[row,t]=(float(R[row,t])-x_min)/x_length
    print(R)
    f_out = open(f_out,'w')
    for i in range(0,rows):
        f_out.write(str((R[i,0])))
        for j in range(1,cols):
            f_out.write(',%.7f'%(R[i,j]))
        f_out.write('\n')

def std_normalization(f_in='number_age_col71.csv',f_out='number_age_col71_stdtran.csv',rows=1277,cols=74):
    f_in = open(f_in)
    R = np.mat(np.zeros((rows, cols)))
    for i, line in enumerate(f_in):
        segs = line.strip().split(',')[0:]
        for k in range(0, len(segs)):
            R[i, k] = float(segs[k])
    for t in range(1, cols):
        col_t = R[:, t]
        x_means = float(col_t.mean(axis = 0))
        x_std = float(col_t.std(axis=0))
        if x_std !=0:
            for row in range(0,rows):
                R[row,t]=(float(R[row,t]-x_means))/(x_std)
    print(R)
    f_out = open(f_out, 'w')
    for i in range(0, rows):
        f_out.write(str((R[i, 0])))
        for j in range(1, cols):
            f_out.write(',%.7f' % (R[i, j]))
        f_out.write('\n')

def get_temp_data(f_in ='wholetiwen.csv',f_out='5_day_temp.csv',mode='w'):
    csv_reader = csv.reader(open(f_in))
    f = open(f_out, mode)
    x0 = []
    x1 = []
    x2 = []
    for i in csv_reader:
        x0.append(i[0])
        x1.append(i[1])
        date_time = datetime.strptime(i[2], "%Y/%m/%d %H:%M")
        x2.append(date_time)
    print(x2[0])

    # process the first len(x0)-1 lines data
    user_id = x0[0]
    start_time =x2[0]
    f.write(x0[0])
    f.write(' %s'%(x1[0]))
    for k in range(0,len(x0)-1):
        if x0[k]==x0[k+1]:
            time = (x2[k+1] -start_time ).days
            if time <5:
                f.write(' %s'%(x1[k+1]))
        else:
            user_id = x0[k+1]
            start_time=x2[k+1]
            f.write('\n%s' % (x0[k+1]))
            f.write(' %s' % (x1[k+1]))
    f.write('\n')
    f.close()



def get_all_tiwen_data(f_in ='wholetiwen.csv',f_out='all_tiwen_data.dat',mode='w'):
    csv_reader = csv.reader(open(f_in))
    f=open(f_out,mode=mode)
    x0 = []
    x1 = []
    for i in csv_reader:
        x0.append(i[0])
        x1.append(i[1])

    user_id = x0[0]
    f.write(x0[0])
    f.write(' %s' % (x1[0]))

    for k in range(0, len(x0) - 1):
        if x0[k] == x0[k + 1]:
            f.write(' %s' % (x1[k + 1]))
        else:
            user_id = x0[k+1]
            f.write('\n%s' % (x0[k+1]))
            f.write(' %s' % (x1[k+1]))
    f.write('\n')
    f.close()





if __name__=="__main__":
    print("main")


    # getnumber_category()

    # splittwocol('number_category.csv','Number.csv','Category.csv','w')
    # get_number_age('wholebasic.csv','number_age.csv','w')
    # add_data_fun('number_age.csv',1,'number.csv','number_age_1.csv','w')
    #
    # invoke_fun('xuechangguicheck.csv', '血小板分布宽度', 'col1.csv', 'w', 'Number.csv', 'col1tran.csv','number_age_1.csv', 'number_age_col1.csv')
    # invoke_fun('xuechangguicheck.csv', '嗜碱细胞绝对值', 'col3.csv', 'w', 'Number.csv', 'col3tran.csv','number_age_col1.csv', 'number_age_col3.csv')
    # invoke_fun('xuechangguicheck.csv', '红细胞总数', 'col5.csv', 'w', 'Number.csv', 'col5tran.csv','number_age_col3.csv', 'number_age_col5.csv')
    # invoke_fun('xuechangguicheck.csv', '紅細胞分布宽度标准差', 'col7.csv', 'w', 'Number.csv', 'col7tran.csv', 'number_age_col5.csv', 'number_age_col7.csv')
    # invoke_fun('xuechangguicheck.csv', '平均红细胞体积', 'col9.csv', 'w', 'Number.csv', 'col9tran.csv','number_age_col7.csv', 'number_age_col9.csv')
    # invoke_fun('xuechangguicheck.csv', '中性细胞绝对值', 'col11.csv', 'w', 'Number.csv', 'col11tran.csv', 'number_age_col9.csv', 'number_age_col11.csv')
    # invoke_fun('xuechangguicheck.csv', '平均红细胞血红蛋白浓度', 'col13.csv', 'w', 'Number.csv', 'col13tran.csv', 'number_age_col11.csv', 'number_age_col13.csv')
    #
    # invoke_fun('xuechangguicheck.csv','血小板总数','col15.csv','w','Number.csv','col15tran.csv','number_age_col13.csv','number_age_col15.csv')
    # invoke_fun('xuechangguicheck.csv','血红蛋白','col17.csv','w','Number.csv','col17tran.csv','number_age_col15.csv','number_age_col17.csv')
    # invoke_fun('xuechangguicheck.csv','淋巴细胞绝对值','col19.csv','w','Number.csv','col19tran.csv','number_age_col17.csv','number_age_col19.csv')
    # invoke_fun('xuechangguicheck.csv','白细胞总数','col21.csv','w','Number.csv','col21tran.csv','number_age_col19.csv','number_age_col21.csv')
    # invoke_fun('xuechangguicheck.csv','单核细胞百分比','col23.csv','w','Number.csv','col23tran.csv','number_age_col21.csv','number_age_col23.csv')
    # invoke_fun('xuechangguicheck.csv','单核细胞绝对值','col25.csv','w','Number.csv','col25tran.csv','number_age_col23.csv','number_age_col25.csv')
    # invoke_fun('xuechangguicheck.csv','淋巴细胞百分比','col27.csv','w','Number.csv','col27tran.csv','number_age_col25.csv','number_age_col27.csv')
    # invoke_fun('xuechangguicheck.csv','嗜碱细胞百分比','col29.csv','w','Number.csv','col29tran.csv','number_age_col27.csv','number_age_col29.csv')
    # invoke_fun('xuechangguicheck.csv','嗜酸细胞百分比','col31.csv','w','Number.csv','col31tran.csv','number_age_col29.csv','number_age_col31.csv')
    # invoke_fun('xuechangguicheck.csv','血小板压积','col33.csv','w','Number.csv','col33tran.csv','number_age_col31.csv','number_age_col33.csv')
    # invoke_fun('xuechangguicheck.csv','大血小板比率','col35.csv','w','Number.csv','col35tran.csv','number_age_col33.csv','number_age_col35.csv')
    # invoke_fun('xuechangguicheck.csv','红细胞压积','col37.csv','w','Number.csv','col37tran.csv','number_age_col35.csv','number_age_col37.csv')
    # invoke_fun('xuechangguicheck.csv','红细胞分布宽度变异系数','col39.csv','w','Number.csv','col39tran.csv','number_age_col37.csv','number_age_col39.csv')
    # invoke_fun('xuechangguicheck.csv','中性细胞百分比','col41.csv','w','Number.csv','col41tran.csv','number_age_col39.csv','number_age_col41.csv')
    # invoke_fun('xuechangguicheck.csv','平均红细胞血红蛋白含量','col43.csv','w','Number.csv','col43tran.csv','number_age_col41.csv','number_age_col43.csv')
    # invoke_fun('xuechangguicheck.csv','嗜酸细胞绝对值','col45.csv','w','Number.csv','col45tran.csv','number_age_col43.csv','number_age_col45.csv')
    # invoke_fun('xuechangguicheck.csv','血小板平均体积','col47.csv','w','Number.csv','col47tran.csv','number_age_col45.csv','number_age_col47.csv')
    # invoke_fun('xuechangguicheck.csv','C反应蛋白','col49.csv','w','Number.csv','col49tran.csv','number_age_col47.csv','number_age_col49.csv')


    # invoke_fun('niaoyefenxicheck.csv', '红细胞(镜检)', 'col51.csv', 'w', 'Number.csv', 'col51tran.csv', 'number_age_col49.csv','number_age_col51.csv')
    # invoke_fun('niaoyefenxicheck.csv', '白细胞(镜检)', 'col53.csv', 'w', 'Number.csv', 'col53tran.csv', 'number_age_col51.csv','number_age_col53.csv')
    # invoke_fun('niaoyefenxicheck.csv', '比重', 'col55.csv', 'w', 'Number.csv', 'col55tran.csv', 'number_age_col53.csv','number_age_col55.csv')
    # invoke_fun('niaoyefenxicheck.csv', 'PH值', 'col57.csv', 'w', 'Number.csv', 'col57tran.csv', 'number_age_col55.csv','number_age_col57.csv')


    #
    # invoke_fun('shenghuafenxicheck.csv', '白蛋白', 'col51.csv', 'w', 'Number.csv', 'col51tran.csv', 'number_age_col49.csv','number_age_col51.csv')
    # invoke_fun('shenghuafenxicheck.csv', '丙氨酸氨基转移酶', 'col53.csv', 'w', 'Number.csv', 'col53tran.csv', 'number_age_col51.csv','number_age_col53.csv')
    # invoke_fun('shenghuafenxicheck.csv', '天门冬氨酸氨基转移酶', 'col55.csv', 'w', 'Number.csv', 'col55tran.csv', 'number_age_col53.csv','number_age_col55.csv')


    # add_tiwen_fun('tiwencheck.csv', 'number_age_col71.csv', 'number_age_col71_tiwen.csv', 'w')

    # add_tiwen_fun('tiwen.csv', 'Number.csv', 'tiwencheck.csv', 'w')

    # kmeans(4)

    # merge_tiwen_number_fun()

    # get_all_tiwen_data()
    # merge_tiwen_number_fun('all_tiwen_data.dat','Number.csv','all_tiwen_data_check.csv','w')

    # xman_xmix_normalization()

    # std_normalization()
