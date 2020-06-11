import csv
from datetime import datetime
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
        if (x7 != '') or (x4 == '' and x5 != '' and x6 != '') or (x4 != '' and x5 == '' and x6 != '') or (
                            x4 != '' and x5 != '' and x6 == '') or (x4 != '' and x5 != '' and x6 != ''):
            f.write(x1 + ',4' + '\n')
            continue
        if x8 != '':
            f.write(x1 + ',5' + '\n')
            continue
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
                if 54.00>=float(i[4])>=38.00:
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
    value = [i[2] for i in csv_reader]
    count_value =0
    for k in range (0,len(value)):
        count_value+=float(value[k])
    return '%.2f'%(count_value/len(value))


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


if __name__=="__main__":
    print("main")
    # getnumber_category()
    # delete the file number_category.csv from line 1262 to 1577
    # what's more 10797 10801 10809 10810 10811 10820
    # splittwocol('number_category.csv','Number.csv','Category.csv','w')
    # get_number_age('wholebasic.csv','number_age.csv','w')
    # add_data_fun('number_age.csv',1,'number.csv','Number_Age1.csv','w')

    # invoke_fun('xuechangguicheck.csv', '血小板分布宽度', 'col1.csv', 'w', 'Number.csv', 'col1tran.csv',
    #            'Number_Age1.csv', 'number_age_col1.csv')
    # invoke_fun('xuechangguicheck.csv', '嗜碱细胞绝对值', 'col3.csv', 'w', 'Number.csv', 'col3tran.csv',
    #            'number_age_col1.csv', 'number_age_col3.csv')
    # invoke_fun('xuechangguicheck.csv', '红细胞总数', 'col5.csv', 'w', 'Number.csv', 'col5tran.csv',
    #            'number_age_col3.csv', 'number_age_col5.csv')
    # invoke_fun('xuechangguicheck.csv', '紅細胞分布宽度标准差', 'col7.csv', 'w', 'Number.csv', 'col7tran.csv',
    #            'number_age_col5.csv', 'number_age_col7.csv')
    # invoke_fun('xuechangguicheck.csv', '平均红细胞体积', 'col9.csv', 'w', 'Number.csv', 'col9tran.csv',
    #            'number_age_col7.csv', 'number_age_col9.csv')
    # invoke_fun('xuechangguicheck.csv', '中性细胞绝对值', 'col11.csv', 'w', 'Number.csv', 'col11tran.csv',
    #            'number_age_col9.csv', 'number_age_col11.csv')
    # invoke_fun('xuechangguicheck.csv', '平均红细胞血红蛋白浓度', 'col13.csv', 'w', 'Number.csv', 'col13tran.csv',
    #            'number_age_col11.csv', 'number_age_col13.csv')
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
    # invoke_fun('niaoyefenxicheck.csv', '红细胞(镜检)', 'col58.csv', 'w', 'Number.csv', 'col58tran.csv', 'number_age_col57.csv','number_age_col58.csv')
    # invoke_fun('niaoyefenxicheck.csv', '白细胞(镜检)', 'col60.csv', 'w', 'Number.csv', 'col60tran.csv', 'number_age_col58.csv','number_age_col60.csv')
    # invoke_fun('niaoyefenxicheck.csv', '比重', 'col62.csv', 'w', 'Number.csv', 'col62tran.csv', 'number_age_col60.csv','number_age_col62.csv')
    # invoke_fun('niaoyefenxicheck.csv', 'PH值', 'col64.csv', 'w', 'Number.csv', 'col64tran.csv', 'number_age_col62.csv','number_age_col64.csv')
    # invoke_fun('shenghuafenxicheck.csv', '白蛋白', 'col66.csv', 'w', 'Number.csv', 'col66tran.csv', 'number_age_col64.csv','number_age_col66.csv')
    # invoke_fun('shenghuafenxicheck.csv', '丙氨酸氨基转移酶', 'col68.csv', 'w', 'Number.csv', 'col68tran.csv', 'number_age_col66.csv','number_age_col68.csv')
    # invoke_fun('shenghuafenxicheck.csv', '天门冬氨酸氨基转移酶', 'col70.csv', 'w', 'Number.csv', 'col70tran.csv', 'number_age_col68.csv','number_age_col70.csv')
    #
    # add_tiwen_fun('tiwen.csv','number_age_col70.csv','final.csv','w')

    # add_tiwen_fun('tiwen.csv','Number.csv','tiwenprediction.csv','w')
    # add_data_fun('kmeanstiwen.csv',5,'number_age_col70.csv','finaldata70+5.csv','w')
