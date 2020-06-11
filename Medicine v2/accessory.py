import csv
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import numpy as np
import random

def plot_user_tiwen_length():
    f = open('all_tiwen_data_check.csv')
    R=[]
    for i,line in enumerate(f):
        lines = line.strip().split(',')[1:]
        R.append(len(lines))
    R=sorted(R)
    print(R)
    k =np.arange(0,200)
    k_count =[]
    for m in range(0,200):
        for n in range(0,len(R)):
            if R[n]>m:
                k_count.append(n)
                break
    k_count=np.array(k_count)
    print(k_count)

    plt.plot(k,k_count[0:]/(float(len(R))))
    plt.ylabel('proba')
    plt.xlabel('users\' temp count ')
    plt.ylim([0, 1])
    plt.xlim([0, 200])
    plt.show()

# plot_user_tiwen_length()

def tran_tiwencheck(f_in='5_day_tiwencheck.csv',f_out='5_day_50_check.csv',dim=50):
    f_in = open(f_in)
    f_out =open(f_out,'w')
    for i,line in enumerate(f_in):
        temp =line.strip().split(',')[0:]
        f_out.write(str(temp[0]))
        if len(temp)-1<dim:
            for k in range(1,len(temp)):
                f_out.write(',%s'%(temp[k]))
            for j in range(0,dim+1-len(temp)):
                f_out.write(',%s'%(temp[len(temp)-1]))
        else:
            for k in range(1,dim+1):
                f_out.write(',%s'%(temp[k]))
        f_out.write('\n')
    f_in.close()
    f_out.close()

# tran_tiwencheck()
# tran_tiwencheck(f_out='5_day_100_check.csv',dim=100)
# tran_tiwencheck(f_out='5_day_40_check.csv',dim=40)
# tran_tiwencheck(f_out='5_day_25_check.csv',dim=25)
# tran_tiwencheck(f_out='5_day_15_check.csv',dim=15)

def read_file_len(f_in='5_day_50_check.csv'):
    f_in =open(f_in)
    for i,line in enumerate(f_in):
        if i==0:
            lines =line.strip().split(',')[1:]
            print(len(lines))
            break
    f_in.close()

# read_file_len()

def read_tiwen_check(f_in='tiwencheck.csv'):
    f_in =open(f_in)
    R=[]
    for i,line in enumerate(f_in):
        temp=[]
        lines =line.strip().split(',')[1:]
        for k in range(0,len(lines)):
            temp.append(lines[k])
        R.append(temp)
    return R

def reshape_file(f_in='number_category.csv'):
    f_in =open(f_in)
    R=[]
    for i ,line in enumerate(f_in):
        temp=[]
        lines= line.strip().split(',')[0:]
        for k in range(0,len(lines)):
            temp.append(lines[k])
        R.append(temp)
        R.sort(key=lambda x:x[1])
    return R


# print(reshape_file())

def c3_to_c2():
    f_in =open('number_category.csv')
    f_out=open('number_category_2.csv','w')
    for i ,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        print(line[1])
        if line[1]=='3':
            f_out.write(line[0]+',2\n')
        else:
            f_out.write(line[0]+','+line[1]+'\n')

def percentage():
    f_in = open('number_category.csv')
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



# percentage()

def normalization(f_in ='5_day_25_check.csv',f_out='5_day_25_nor.dat'):
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
                f_out.write(' 1.000')
        else:
            for temp_i in temp:
                f_out.write(' %.3f'%((float(temp_i)-min_temp)/(max_temp-min_temp)))
        f_out.write('\n')



def combine_fun(f_in_1='number_age_col71tran.csv',f_in_2='number_daye.dat',f_out='number_age_col73tran.csv'):
    f_in_1 = open(f_in_1)
    f_in_2=open(f_in_2)
    metrix_1 =[]
    metrix_2=[]
    for i,lines in enumerate(f_in_1):
        line =lines.strip().split(',')[0:]
        metrix_1.append(line)
    for k,lines in enumerate(f_in_2):
        line =lines.strip().split(' ')[0:]
        metrix_2.append(line)
    for j in range(0,len(metrix_1)):
        for l in range(0,len(metrix_2)):
            if int(float(metrix_1[j][0]))==int(metrix_2[l][0]):
                for m in range(1,len(metrix_2[l])):
                    metrix_1[j].append(metrix_2[l][m])
                break
    f_out=open(f_out,'w')
    for i in range(0,len(metrix_1)):
        f_out.write('%.1f'%(float(metrix_1[i][0])))
        for j in range(1,len(metrix_1[i])):
            f_out.write(',%.7f'%(float(metrix_1[i][j])))
        f_out.write('\n')


# combine_fun(f_in_1='number_age_col71tran.csv',f_in_2='number_daye.dat',f_out='number_age_col73tran.csv')
# combine_fun(f_in_1='number_age_col73tran.csv',f_in_2='number_month.dat',f_out='number_age_col85tran.csv')




def change_time():
    f_in = open('wholetiwen.csv')
    wholedata = []
    for i, lines in enumerate(f_in):
        line = lines.strip().split(',')[0:]
        wholedata.append(line)
    f_out = open('testoutput.dat', 'w')
    for i in range(0,len(wholedata)):
        time_1 = datetime.strptime(wholedata[i][2], "%Y/%m/%d %H:%M")
        if 0<=time_1.hour<=11:
            f_out.write('%s,%.1f,%s 12:00:00\n' % (wholedata[i][0], float(wholedata[i][1]), time_1.strftime("%Y/%m/%d")))
        else:
            f_out.write('%s,%.1f,%s 0:00:00\n' % (wholedata[i][0], float(wholedata[i][1]), (time_1+timedelta(1)).strftime("%Y/%m/%d")))

# change_time()

def again2():
    f_in =open('testoutput.dat')
    f_out=open('test.dat','w')
    wholedata = []
    for i, lines in enumerate(f_in):
        line = lines.strip().split(',')[0:]
        wholedata.append(line)
    temp =0.0
    count=0
    i=0
    while i <len(wholedata):
        if i==len(wholedata)-1:
            time_1 = datetime.strptime(wholedata[i][2], "%Y/%m/%d %H:%M:%S")
            f_out.write(
                '%s,%.1f,%s\n' % (wholedata[i][0], float(wholedata[i][1]), time_1.strftime("%Y/%m/%d %H:%M:%S")))
            break
        if wholedata[i][0]==wholedata[i+1][0]:
            temp+=float(wholedata[i][1])
            count+=1
            time_1 = datetime.strptime(wholedata[i][2], "%Y/%m/%d %H:%M:%S")
            time_2 = datetime.strptime(wholedata[i + 1][2], "%Y/%m/%d %H:%M:%S")
            if time_1 != time_2:
                f_out.write(
                    '%s,%.1f,%s\n' % (wholedata[i][0], float(wholedata[i][1]), time_1.strftime("%Y/%m/%d %H:%M:%S")))
                i+=1
                count=0
                temp=0.0
                continue
        else:
            time_1 = datetime.strptime(wholedata[i][2], "%Y/%m/%d %H:%M:%S")
            f_out.write(
                '%s,%.1f,%s\n' % (wholedata[i][0], float(wholedata[i][1]), time_1.strftime("%Y/%m/%d %H:%M:%S")))
            i+=1
            count = 0
            temp = 0.0
            continue
        while wholedata[i][0]==wholedata[i+1][0]:
            time_1 = datetime.strptime(wholedata[i][2], "%Y/%m/%d %H:%M:%S")
            time_2 = datetime.strptime(wholedata[i + 1][2], "%Y/%m/%d %H:%M:%S")
            if time_1!=time_2:
                f_out.write(
                    '%s,%.1f,%s\n' % (wholedata[i][0], float(temp/count), time_1.strftime("%Y/%m/%d %H:%M:%S")))
                temp=0.0
                count=0
                break
            else:
                temp+=float(wholedata[i+1][1])
                count+=1
            if i!=len(wholedata)-2:
                i+=1
        i+=1

# again2()

def process_test_dat(f_in='test.dat',f_number ='number.csv',f_out='5_day_10_unmerged.dat'):
    csv_reader2 = csv.reader(open(f_number))
    number = [k[0] for k in csv_reader2]
    f_in =open(f_in)
    f_out=open(f_out,'w')
    wholedata=[]
    for i,lines in enumerate(f_in):
        line=lines.strip().split(',')[0:]
        wholedata.append(line)

    f_out.write(wholedata[0][0])
    f_out.write(',%s'%(wholedata[0][1]))
    start_time = datetime.strptime(wholedata[0][2], "%Y/%m/%d %H:%M:%S")
    for k in range(0,len(wholedata)-1):
        time_2 = datetime.strptime(wholedata[k + 1][2], "%Y/%m/%d %H:%M:%S")
        if wholedata[k][0]==wholedata[k+1][0]:
            time = (time_2 - start_time).days
            if time<5:
                f_out.write(',%s'%(wholedata[k+1][1]))
        else:
            start_time =time_2
            f_out.write('\n%s'%(wholedata[k+1][0]))
            f_out.write(',%s'%(wholedata[k+1][1]))


def merge_temp_fun(f_number ='number.csv',f_in ='5_day_10_unmerged.dat',f_out='5_day_10_check.dat'):
    csv_reader2 = csv.reader(open(f_number))
    number = [k[0] for k in csv_reader2]
    f_out=open(f_out,'w')
    f_in=open(f_in)
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        if line[0] in number:
            f_out.write(line[0])
            for k in line[1:]:
                f_out.write(',%s'%(k))
            f_out.write('\n')



# process_test_dat()
# merge_temp_fun()
# tran_tiwencheck(f_in ='5_day_10_check.dat',f_out='5_day_10_equal_length.csv',dim=10)
# normalization(f_in='5_day_10_equal_length.csv',f_out='5_day_10_nor.dat')


def tiwen_minus_37():
    f_in =open('5_day_15_check.csv')
    f_out=open('5_day_15_minus_37.csv','w')
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        f_out.write(line[0])
        for k in line[1:]:
            f_out.write(',%.1f'%(float(k)-37.0))
        f_out.write('\n')

# tiwen_minus_37()

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

split_train_validation_test_data()
