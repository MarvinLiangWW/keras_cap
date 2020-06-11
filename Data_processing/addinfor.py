# import csv
# csv_reader = csv.reader(open('huizongnumberage.csv'))
# number=[]
# age=[]
# for k in csv_reader:
#     number.append(k[0])
#     age.append(k[1])
#
#
# csv_reader2 = csv.reader(open('number.csv'))
# f = open('temp.csv', 'w+')
# for i in csv_reader2:
#     x = i[0]
#     f.write(x)
#     lennumber = len(number)
#     a=0
#     while a<lennumber:
#         if number[a] == x:
#             f.write(','+age[a])
#             break
#         a += 1
#     f.write('\n')
# f.close()

import csv

def adddatafun(filename1,filename2,output,mode):
    csv_reader = csv.reader(open(filename1))
    csv_reader2 = csv.reader(open(filename2))
    number=[]
    parameter=[]
    for i in csv_reader:
        number.append(i[0])
        parameter.append(i[1])
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
                f.write(','+parameter[a])
                break
            a+=1
        if a == lennumber:
            f.write(','+ '0')
        f.write('\n')
    f.close()


def addtiwenfun(filename1,filename2,output,mode):
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

def quchongfun(filename, output,mode):
    csv_reader = csv.reader(open(filename))
    number=[]
    parameter =[]
    for i in csv_reader:
        number.append(i[0])
        parameter.append(i[1])
    f=open(output,mode)
    print(number)
    print(parameter)
    a=0
    while a<len(number):
        b=a
        count =float(parameter[a])
        if a == len(number)-1:
            f.write(number[a]+','+ '%.2f'%(float(parameter[a]))+'\n')
            a+=1
        else:
            while number[a+1]==number[a]:
                a+=1
                count += float(parameter[a])
                if a==len(number):
                    break
            #print(count)
            c='%.2f'% (count/(a-b+1))
            f.write(number[a]+','+str(c)+'\n')
            a+=1
    f.close()




def devidetwocol(filename,outputcol1,ouputcol2,mode):
    csv_reader = csv.reader(open(filename))
    f1=open(outputcol1,mode)
    f2 = open(ouputcol2,mode)
    for i in csv_reader:
        f1.write(i[0]+'\n')
        f2.write(i[1]+'\n')
    f1.close()
    f2.close()


def getnumber_category():
    csv_reader1 = csv.reader(open('2015-10-5zong.csv', encoding='GBK'))
    '''GBK的原因：支（衣）原体肺炎在数据中有‘1’和‘衣原体’两种标识'''
    f = open('number-category.csv', 'w+')
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



# getnumber_category()
# delete the file number-category.csv from line 1262 to 1577
# what's more 10797 10801 10809 10810 10811 10820


devidetwocol('number-category.csv','number.csv','category.csv','w')
adddatafun('huizongnumberage.csv','number.csv','number_age.csv','w')

adddatafun('white.csv','number_age.csv','number_age_white.csv','w')
quchongfun('bxbzs.csv','bxbzsxiugai.csv','w')
adddatafun('bxbzsxiugai.csv','number_age_white.csv','number_age_white_bxbzs.csv','w')
quchongfun('zxlxbbfb.csv','zxlxbbfbxiugai.csv','w')
adddatafun('zxlxbbfbxiugai.csv','number_age_white_bxbzs.csv','number_age_white_bxbzs_zxlxbbfb.csv','w')

quchongfun('lbxbbfb.csv','lbxbbfbxiugai.csv','w')

adddatafun('lbxbbfbxiugai.csv','number_age_white_bxbzs_zxlxbbfb.csv','number_age_white_bxbzs_zxlxbbfb_lbxbbfb.csv','w')
quchongfun('cfydb.csv','cfydbxiugai.csv','w')
adddatafun('cfydbxiugai.csv','number_age_white_bxbzs_zxlxbbfb_lbxbbfb.csv','number_age_white_bxbzs_zxlxbbfb_lbxbbfb_cfydb.csv','w')

quchongfun('basajzym.csv','basajzymxiugai.csv','w')
adddatafun('basajzymxiugai.csv','number_age_white_bxbzs_zxlxbbfb_lbxbbfb_cfydb.csv','number_age_white_bxbzs_zxlxbbfb_lbxbbfb_cfydb_ALT.csv','w')

quchongfun('tmdasajzym.csv','tmdasajzymxiugai.csv','w')
adddatafun('tmdasajzymxiugai.csv','number_age_white_bxbzs_zxlxbbfb_lbxbbfb_cfydb_ALT.csv','number_age_white_bxbzs_zxlxbbfb_lbxbbfb_cfydb_ALT_AST.csv','w')


addtiwenfun('tiwen.csv','number_age_white_bxbzs_zxlxbbfb_lbxbbfb_cfydb_ALT_AST.csv','number_age_white_bxbzs_zxlxbbfb_lbxbbfb_cfydb_ALT_AST_tiwen.csv','w')


