import csv
from datetime import datetime
import random


def split_wholejichu():
    csv_reader = csv.reader(open('wholejichu.csv'))
    # f1 =open('xuechanggui.csv','w')
    # f2 =open('niaoyefenxi.csv','w')
    f3 =open('shenghuafenxi.csv','w')
    for i in csv_reader:
        x2 = i[2]
        # if x2 =='血常规':
        #     f1.write(i[0])
        #     t=1
        #     while t<len(i):
        #         f1.write(','+i[t])
        #         t+=1
        #     f1.write('\n')
        # if x2 =='尿液分析':
        #     f2.write(i[0])
        #     t = 1
        #     while t < len(i):
        #         f2.write(',' + i[t])
        #         t += 1
        #     f2.write('\n')
        if x2 == '生化':
            f3.write(i[0])
            t = 1
            while t < len(i):
                f3.write(',' + i[t])
                t += 1
            f3.write('\n')
    # f1.close()
    # f2.close()
    f3.close()


def filtersort(filename,filename2,output,mode):
    csv_reader = csv.reader(open(filename))
    x0 =[]
    x1 =[]
    x2 =[]
    x3 =[]
    x4 =[]
    x5 =[]
    for i in csv_reader:
        x0.append(i[0])
        x1.append(i[1])
        x2.append(i[2])
        x3.append(i[3])
        x4.append(i[4])
        x5.append(i[5])
    csv_readnumber = csv.reader(open(filename2))
    number = [i[0] for i in csv_readnumber]
    f = open(output, mode)
    for k in range(0,len(number)):
        defaulttime = datetime.strptime('2100/1/1 12:12', "%Y/%m/%d %H:%M")
        for j in range(0,len(x0)):
            time = datetime.strptime(x1[j], "%Y/%m/%d %H:%M")
            if number[k] == x0[j] and time<=defaulttime:
                f.write(x0[j])
                f.write(','+x1[j])
                f.write(','+x2[j])
                f.write(','+x3[j])
                f.write(','+x4[j])
                f.write(','+x5[j])
                f.write('\n')
                defaulttime =time


def fun(filename,testname,output,mode):
    csv_reader = csv.reader(open(filename))
    f = open(output, mode)
    for i in csv_reader:
        if i[3] ==testname:
            # f.write(i[0])
            if i[5] =='阴性':
                if i[4] =='阴性':
                    f.write(i[0] + ',1\n')
                else:
                    f.write(i[0]+',0\n')
            else:
                s = i[5].split('～')
                s0 = '%.2f' % (float(s[0]))
                s1 = '%.2f' % (float(s[1]))
                try:
                    if float(s1) >= float(i[4]) >= float(s0):
                        f.write(i[0] + ',' + i[4] + ',1\n')
                    else:
                        f.write(i[0] + ',' + i[4] + ',0\n')
                except ValueError:
                    f.write('')

    f.close()

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
            f.write(','+str(random.randint(0,1)))
            # pretend that the average value of test is always normal
        f.write('\n')

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


def invoke_fun(dataset,testname,output,mode,number,outputtran,lastoutput,finaloutput):
    fun(dataset, testname, output, mode)
    add0fun(output, number, 1, outputtran, mode)
    add_data_fun(outputtran, 1, lastoutput, finaloutput, mode)


if __name__=="__main__":
    print("main")
    # split_wholejichu()
    # filtersort('xuechanggui.csv','Number.csv','xuechangguicheck.csv','w')
    # filtersort('niaoyefenxi.csv','Number.csv','niaoyefenxicheck.csv','w')
    # filtersort('shenghuafenxi.csv','Number.csv','shenghuafenxicheck.csv','w')

    # fun('niaoyefenxicheck.csv','酮体', 'col50.csv', 'w')
    # add0fun('col50.csv','Number.csv',1,'col50tran.csv','w')
    # add_data_fun('col50tran.csv',1,'number_age_col49.csv','number_age_col50.csv','w')
    #
    # invoke_fun('niaoyefenxicheck.csv','潜血', 'col51.csv', 'w','Number.csv','col51tran.csv','number_age_col50.csv','number_age_col51.csv')
    # invoke_fun('niaoyefenxicheck.csv','葡萄糖', 'col52.csv', 'w','Number.csv','col52tran.csv','number_age_col51.csv','number_age_col52.csv')
    # invoke_fun('niaoyefenxicheck.csv','尿胆原', 'col53.csv', 'w','Number.csv','col53tran.csv','number_age_col52.csv','number_age_col53.csv')
    # invoke_fun('niaoyefenxicheck.csv','胆红素', 'col54.csv', 'w','Number.csv','col54tran.csv','number_age_col53.csv','number_age_col54.csv')
    # invoke_fun('niaoyefenxicheck.csv','白细胞酯酶', 'col55.csv', 'w','Number.csv','col55tran.csv','number_age_col54.csv','number_age_col55.csv')
    # invoke_fun('niaoyefenxicheck.csv','尿蛋白', 'col56.csv', 'w','Number.csv','col56tran.csv','number_age_col55.csv','number_age_col56.csv')
    # invoke_fun('niaoyefenxicheck.csv','亚硝酸盐', 'col57.csv', 'w','Number.csv','col57tran.csv','number_age_col56.csv','number_age_col57.csv')






