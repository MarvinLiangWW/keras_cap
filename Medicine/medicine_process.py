import csv
from datetime import datetime,timedelta
import numpy as np
from matplotlib import pyplot as plt
#result: get 316 medicines
def read_wholeyizhu(f_in='wholeyizhu_check.csv',f_out='medicine_list.dat',col=2):
    csv_reader =csv.reader(open(f_in))
    medicine =[]
    for i in csv_reader:
        if csv_reader.line_num  ==1:
            continue
        medicine_temp = i[col]
        if medicine_temp in medicine:
            continue
        else:
            medicine.append(medicine_temp)
    medicine_list =open(f_out,'w')
    # medicine_list.write("[none]none\n")
    for k in range(0,len(medicine)):
        medicine_list.write(medicine[k]+'\n')
    medicine_list.close()

def tiwen_file_to_list(f_temp='wholetiwen.csv'):
    read_temp =csv.reader(open(f_temp))
    tiwen_list =[]
    for i in read_temp:
        temp_list=[]
        temp_list.append(i[0])
        temp_list.append(i[1])
        temp_list.append(i[2])
        tiwen_list.append(temp_list)
    return tiwen_list

def yizhu_file_to_list(f_medicine='wholeyizhu_check.csv'):
    read_medicine =csv.reader(open(f_medicine))
    medicine_list=[]
    for i in read_medicine:
        if read_medicine.line_num  ==1:
            continue
        temp_list=[]
        temp_list.append(i[0])
        temp_list.append(i[2])
        temp_list.append(i[3])
        temp_list.append(i[4])
        temp_list.append(i[5])
        medicine_list.append(temp_list)
    return medicine_list

def read_medicine_list(f_in='medicine_list.dat'):
    f=open(f_in)
    medicine_name_list=[]
    for i,line in enumerate(f):
        medicine_name = line.strip()
        medicine_name_list.append(medicine_name)
    return medicine_name_list


def function_a():
    temp_list =tiwen_file_to_list()
    medicine_list =yizhu_file_to_list()
    medicine_name_list = read_medicine_list()
    # print(medicine_list[0][3])

    for i in range(0,len(temp_list)):
        #get temp time
        temp_time = datetime.strptime(temp_list[i][2], "%Y/%m/%d %H:%M")
        for j in range(0,len(medicine_list)):
            if medicine_list[j][0]!=temp_list[i][0]:
                continue
            #get medicine_begin_time
            medicine_begin_time =datetime.strptime(medicine_list[j][2],"%Y/%m/%d %H:%M")

            #get medicine_end_time
            if medicine_list[j][3]!='':
                medicine_end_time =datetime.strptime(medicine_list[j][3],"%Y/%m/%d %H:%M")
            else:
                medicine_end_time=medicine_begin_time

            #get frequence
            gap_sequence =medicine_list[j][4]
            if gap_sequence=='每天1次'or '每晚1次':
                gap_time=24
            elif gap_sequence=='每天2次' or '每12小时1次' or '每12小时一次':
                gap_time=12
            elif gap_sequence=='每8小时1次'or'每天3次' or '每8小时一次':
                gap_time=8
            elif gap_sequence=='每6小时1次'or'每6小时一次'or'每天4次':
                gap_time=6
            elif gap_sequence=='每5小时1次':
                gap_time=5
            elif gap_sequence=='每4小时1次'or'每8小时2次':
                gap_time=4
            elif gap_sequence=='每3小时一次':
                gap_time=3
            elif gap_sequence=='每2小时1次':
                gap_time=2
            elif gap_sequence=='立即':
                gap_time=1
            elif gap_sequence=='每周2次':
                gap_time=84
            elif gap_sequence=='间隔'or'每隔1天':
                gap_time=48
            elif gap_sequence=='每周1次':
                gap_time=168

            while medicine_begin_time<=medicine_end_time:
                # if i == 0 or temp_list[i - 1][0] != temp_list[i][0]:
                #     if medicine_begin_time < temp_time and medicine_name_list.index(medicine_list[j][1]) + 1 not in temp_list[i]:
                #         temp_list[i].append(medicine_name_list.index(medicine_list[j][1]) + 1)
                # else:
                if temp_time-timedelta(minutes=60) <= medicine_begin_time <= temp_time:
                    if medicine_name_list.index(medicine_list[j][1]) + 1 not in temp_list[i]:
                        temp_list[i].append(medicine_name_list.index(medicine_list[j][1]) + 1)
                medicine_begin_time=medicine_begin_time+timedelta(hours=gap_time)
        print(temp_list[i])


def function_b():
    temp_list = tiwen_file_to_list()
    medicine_list = yizhu_file_to_list()
    medicine_name_list = read_medicine_list()

    for i in range(0,len(medicine_list)):
    # for i in range(0,50):
        if i%1000==0:
            print(i)

        # get medicine_begin_time
        medicine_begin_time = datetime.strptime(medicine_list[i][2], "%Y/%m/%d %H:%M")

        # get medicine_end_time
        if medicine_list[i][3] != '':
            medicine_end_time = datetime.strptime(medicine_list[i][3], "%Y/%m/%d %H:%M")
        else:
            medicine_end_time = medicine_begin_time

        # get frequence
        gap_sequence = medicine_list[i][4]
        if gap_sequence == '每天1次' or '每晚1次':
            gap_time = 24
        elif gap_sequence == '每天2次' or '每12小时1次' or '每12小时一次':
            gap_time = 12
        elif gap_sequence == '每8小时1次' or '每天3次' or '每8小时一次':
            gap_time = 8
        elif gap_sequence == '每6小时1次' or '每6小时一次' or '每天4次':
            gap_time = 6
        elif gap_sequence == '每5小时1次':
            gap_time = 5
        elif gap_sequence == '每4小时1次' or '每8小时2次':
            gap_time = 4
        elif gap_sequence == '每3小时一次':
            gap_time = 3
        elif gap_sequence == '每2小时1次':
            gap_time = 2
        elif gap_sequence == '立即':
            gap_time = 1
        elif gap_sequence == '每周2次':
            gap_time = 84
        elif gap_sequence == '间隔' or '每隔1天':
            gap_time = 48
        elif gap_sequence == '每周1次':
            gap_time = 168

        while medicine_begin_time <= medicine_end_time:
            for j in range(0, len(temp_list)):
                if medicine_list[i][0] != temp_list[j][0]:
                    continue
                temp_time = datetime.strptime(temp_list[j][2], "%Y/%m/%d %H:%M")
                if temp_time-timedelta(minutes=180) <= medicine_begin_time <= temp_time :
                    if medicine_name_list.index(medicine_list[i][1]) + 1 not in temp_list[j]:
                        temp_list[j].append(medicine_name_list.index(medicine_list[i][1]) + 1)
                    break
            medicine_begin_time=medicine_begin_time+timedelta(hours=gap_time)

    f_out=open('tiwen_yongyao_infor.dat','w')
    for i in range(0, len(temp_list)):
        for k in range(0,len(temp_list[i])):
            f_out.write('%s '%str(temp_list[i][k]))
        f_out.write('\n')
    f_out.close()


def period_time():
    temp_list=tiwen_file_to_list()
    f_out=open('tiwen_period_time.dat','w')
    for i in range(0,len(temp_list)-1):
        if temp_list[i][0]==temp_list[i+1][0]:
            temp_time = datetime.strptime(temp_list[i][2], "%Y/%m/%d %H:%M")
            temp_time2 =datetime.strptime(temp_list[i+1][2],"%Y/%m/%d %H:%M")
            f_out.write("%s\n"%(temp_time2-temp_time))


def analyse_period_time(f_in='tiwen_period_time.dat'):
    f_in=open(f_in)
    array =np.zeros((24))
    for i,lines in enumerate(f_in):
        line =lines.strip()
        try:
            temp_time = datetime.strptime(line, "%H:%M:%S")
        except ValueError:
            continue

        array[temp_time.hour]+=1
    x =range(0,24)
    y =sum(array)
    # for i in range(1,24):
    #     array[i]+=array[i-1]

    print(array)
    print(y)
    plt.plot(x,array[x]/y)
    plt.show()






if __name__=='__main__':
    print("main")
    # function_a()
    function_b()
    # read_wholeyizhu()
    # read_wholeyizhu(f_out='medicine_frequent.dat',col =5)
