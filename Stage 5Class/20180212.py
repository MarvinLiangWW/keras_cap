import numpy as np
import pandas as pd
from datetime import datetime,timedelta
import time

def temp2dict(filename,file_output):
    dict={}
    with open(filename,'r') as f_temp:
        for i,lines in enumerate(f_temp):
            line =lines.strip().split(',')
            line[1] =float(line[1])
            if line[1] <=37.2:
                line[1] =0.0
            elif 37.2<=line[1]<=40.0:
                line[1] =(line[1]-37.2)/2.8
            else:
                line[1] =1.0
            line[2] =datetime.strptime(line[2],"%Y/%m/%d %H:%M")
            if line[0] in dict:
                dict[line[0]].append([line[1],line[2]])
            else:
                dict[line[0]] =[[line[1],line[2]]]
    new_dict={}
    for key in dict.keys():
        time = dict[key][0][1]
        for i,item in enumerate(dict[key]):
            if 0<=(item[1]-time).total_seconds()/60<=48*60:
                if key in new_dict:
                    new_dict[key].append(item[0])
                else:
                    new_dict[key]=[item[0]]
            else:
                pass
    print('new_dict.shape',len(new_dict))

    # filter temp sequence less than 3 and at least one temp larger than 37.2
    new2dict={}
    for key in new_dict.keys():
        if len(new_dict[key])<=3:
            pass
        elif sum(new_dict[key]) ==0.0:
            pass
        else:
            new2dict[key]=new_dict[key]

    print('new2dict.shape',len(new2dict))

    with open(file_output,'w',encoding='utf-8') as f_out:
        for key in new2dict.keys():
            f_out.write(str(key))
            for item in new2dict[key]:
                f_out.write(',%.7f'%(item))
            f_out.write('\n')


if __name__ =='__main__':
    print('main')
    temp2dict('wholetiwen.csv','temp.csv')