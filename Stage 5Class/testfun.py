import csv
from datetime import datetime
import numpy as np

def split_wholejichu():
    csv_reader = csv.reader(open('wholejichu.csv'))
    f1 =open('xuechanggui.csv','w')
    f2 =open('niaoyefenxi.csv','w')
    for i in csv_reader:
        x2 = i[2]
        if x2 =='血常规':
            f1.write(i[0])
            t=1
            while t<len(i):
                f1.write(','+i[t])
                t+=1
            f1.write('\n')
        if x2 =='尿液分析':
            f2.write(i[0])
            t = 1
            while t < len(i):
                f2.write(',' + i[t])
                t += 1
            f2.write('\n')
    f1.close()
    f2.close()


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

def function_1():
    f_number=open('number_669.csv')
    number=[]
    for i,lines in enumerate(f_number):
        line =lines.strip().split(',')
        number.append(line[0])
    print(number)
    print(type(number[0]))

    f_out_1=open('')


    f_in =open('wholejichu.csv')
    count_tian=0
    count_c=0
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        if i==10:
            print(line[0])
            print(line[3])
        if line[0] in number and line[3]=='天门冬氨酸氨基转移酶':
            count_tian +=1
        if line[0] in number and line[3]=='C反应蛋白':
            count_c +=1


    print('count_tian',count_tian)
    print('count_c',count_c)


def pencentage():
    f_in =open('Category.csv','r')
    penc =[0]*5
    for i,lines in enumerate(f_in):
        line= lines.strip()
        penc[int(line)-1]+=1
    print(penc)
    print('支原体','病毒','细菌','混合','未知')
    print(np.asarray(penc)/sum(penc))


if __name__=="__main__":
    print("main")
    # pencentage()

    # function_1()
    # split_wholejichu()
    # filtersort('wholejichu.csv','testoutput.csv','w')
    # filtersort('xuechanggui.csv','Number.csv','xuechangguicheck.csv','w')
    # filtersort('niaoyefenxi.csv','Number.csv','niaoyefenxicheck.csv','w')
