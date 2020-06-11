import csv
import matplotlib.pyplot as plt

import numpy as np
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


print(reshape_file())

