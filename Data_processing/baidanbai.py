import csv
#csv_reader = csv.reader(open('2014.6.1-2014.8.31jichu.csv'))


def  getTestInfor(filename,testname,output,mode):
    csv_reader = csv.reader(open(filename))
    f = open(output,mode)
    for i in csv_reader:
        x0 = i[0]
        x1 = i[1]
        x2 = i[2]
        x3 = i[3]
        x4 = i[4]
        x5 = i[5]

        if x3 == testname and testname == '白细胞总数':
            if x2 == '血常规':
                f.write(x0 + ',' + x4 + '\n')
            continue
        if x3 == testname and testname == 'C反应蛋白':
            if x4[0]=='<' or x4[0]=='>':
                f.write(x0 + ',' + x4[1:] + '\n')
            else:
                f.write(x0 + ',' + x4 + '\n')
            continue
        if x3 == testname:
            f.write(x0 + ',' + x4 + '\n')


    f.close()

# getTestInfor('2014.6.1-2014.8.31jichu.csv','白蛋白','white.csv','a')

# getTestInfor('2014.6.1-2014.8.31jichu.csv','白蛋白','white.csv','w')
# getTestInfor('2014.9.1-2014.10.31jichu.csv','白蛋白','white.csv','a')
# getTestInfor('2014.11.1-2014.11.31jichu.csv','白蛋白','white.csv','a')
# getTestInfor('2014.12.1-2014.12.31jichu.csv','白蛋白','white.csv','a')
# getTestInfor('2015.1.1-2015.1.31jichu.csv','白蛋白','white.csv','a')
# getTestInfor('2015.2.1-2015.2.28jichu.csv','白蛋白','white.csv','a')
# getTestInfor('2015.3.1-2015.3.31jichu.csv','白蛋白','white.csv','a')
# getTestInfor('2015.4.1-2015.4.31jichu.csv','白蛋白','white.csv','a')
# getTestInfor('2015.5.1-2015.5.31jichu.csv','白蛋白','white.csv','a')

# getTestInfor('2014.6.1-2014.8.31jichu.csv','白细胞总数','bxbzs.csv','w')
# getTestInfor('2014.9.1-2014.10.31jichu.csv','白细胞总数','bxbzs.csv','a')
# getTestInfor('2014.11.1-2014.11.31jichu.csv','白细胞总数','bxbzs.csv','a')
# getTestInfor('2014.12.1-2014.12.31jichu.csv','白细胞总数','bxbzs.csv','a')
# getTestInfor('2015.1.1-2015.1.31jichu.csv','白细胞总数','bxbzs.csv','a')
# getTestInfor('2015.2.1-2015.2.28jichu.csv','白细胞总数','bxbzs.csv','a')
# getTestInfor('2015.3.1-2015.3.31jichu.csv','白细胞总数','bxbzs.csv','a')
# getTestInfor('2015.4.1-2015.4.31jichu.csv','白细胞总数','bxbzs.csv','a')
# getTestInfor('2015.5.1-2015.5.31jichu.csv','白细胞总数','bxbzs.csv','a')

# 中性粒细胞百分比
# getTestInfor('2014.6.1-2014.8.31jichu.csv','中性细胞百分比','zxlxbbfb.csv','w')
# getTestInfor('2014.9.1-2014.10.31jichu.csv','中性细胞百分比','zxlxbbfb.csv','a')
# getTestInfor('2014.11.1-2014.11.31jichu.csv','中性细胞百分比','zxlxbbfb.csv','a')
# getTestInfor('2014.12.1-2014.12.31jichu.csv','中性细胞百分比','zxlxbbfb.csv','a')
# getTestInfor('2015.1.1-2015.1.31jichu.csv','中性细胞百分比','zxlxbbfb.csv','a')
# getTestInfor('2015.2.1-2015.2.28jichu.csv','中性细胞百分比','zxlxbbfb.csv','a')
# getTestInfor('2015.3.1-2015.3.31jichu.csv','中性细胞百分比','zxlxbbfb.csv','a')
# getTestInfor('2015.4.1-2015.4.31jichu.csv','中性细胞百分比','zxlxbbfb.csv','a')
# getTestInfor('2015.5.1-2015.5.31jichu.csv','中性细胞百分比','zxlxbbfb.csv','a')

# #淋巴细胞百分比
# getTestInfor('2014.6.1-2014.8.31jichu.csv','淋巴细胞百分比','lbxbbfb.csv','w')
# getTestInfor('2014.9.1-2014.10.31jichu.csv','淋巴细胞百分比','lbxbbfb.csv','a')
# getTestInfor('2014.11.1-2014.11.31jichu.csv','淋巴细胞百分比','lbxbbfb.csv','a')
# getTestInfor('2014.12.1-2014.12.31jichu.csv','淋巴细胞百分比','lbxbbfb.csv','a')
# getTestInfor('2015.1.1-2015.1.31jichu.csv','淋巴细胞百分比','lbxbbfb.csv','a')
# getTestInfor('2015.2.1-2015.2.28jichu.csv','淋巴细胞百分比','lbxbbfb.csv','a')
# getTestInfor('2015.3.1-2015.3.31jichu.csv','淋巴细胞百分比','lbxbbfb.csv','a')
# getTestInfor('2015.4.1-2015.4.31jichu.csv','淋巴细胞百分比','lbxbbfb.csv','a')
# getTestInfor('2015.5.1-2015.5.31jichu.csv','淋巴细胞百分比','lbxbbfb.csv','a')
#
# # C反应蛋白
# getTestInfor('2014.6.1-2014.8.31jichu.csv','C反应蛋白','cfydb.csv','w')
# getTestInfor('2014.9.1-2014.10.31jichu.csv','C反应蛋白','cfydb.csv','a')
# getTestInfor('2014.11.1-2014.11.31jichu.csv','C反应蛋白','cfydb.csv','a')
# getTestInfor('2014.12.1-2014.12.31jichu.csv','C反应蛋白','cfydb.csv','a')
# getTestInfor('2015.1.1-2015.1.31jichu.csv','C反应蛋白','cfydb.csv','a')
# getTestInfor('2015.2.1-2015.2.28jichu.csv','C反应蛋白','cfydb.csv','a')
# getTestInfor('2015.3.1-2015.3.31jichu.csv','C反应蛋白','cfydb.csv','a')
# getTestInfor('2015.4.1-2015.4.31jichu.csv','C反应蛋白','cfydb.csv','a')
# getTestInfor('2015.5.1-2015.5.31jichu.csv','C反应蛋白','cfydb.csv','a')

# 天门冬氨酸氨基转移酶
# getTestInfor('2014.6.1-2014.8.31jichu.csv','天门冬氨酸氨基转移酶','tmdasajzym.csv','w')
# getTestInfor('2014.9.1-2014.10.31jichu.csv','天门冬氨酸氨基转移酶','tmdasajzym.csv','a')
# getTestInfor('2014.11.1-2014.11.31jichu.csv','天门冬氨酸氨基转移酶','tmdasajzym.csv','a')
# getTestInfor('2014.12.1-2014.12.31jichu.csv','天门冬氨酸氨基转移酶','tmdasajzym.csv','a')
# getTestInfor('2015.1.1-2015.1.31jichu.csv','天门冬氨酸氨基转移酶','tmdasajzym.csv','a')
# getTestInfor('2015.2.1-2015.2.28jichu.csv','天门冬氨酸氨基转移酶','tmdasajzym.csv','a')
# getTestInfor('2015.3.1-2015.3.31jichu.csv','天门冬氨酸氨基转移酶','tmdasajzym.csv','a')
# getTestInfor('2015.4.1-2015.4.31jichu.csv','天门冬氨酸氨基转移酶','tmdasajzym.csv','a')
# getTestInfor('2015.5.1-2015.5.31jichu.csv','天门冬氨酸氨基转移酶','tmdasajzym.csv','a')

# 丙氨酸氨基转移酶
# getTestInfor('2014.6.1-2014.8.31jichu.csv','丙氨酸氨基转移酶','basajzym.csv','w')
# getTestInfor('2014.9.1-2014.10.31jichu.csv','丙氨酸氨基转移酶','basajzym.csv','a')
# getTestInfor('2014.11.1-2014.11.31jichu.csv','丙氨酸氨基转移酶','basajzym.csv','a')
# getTestInfor('2014.12.1-2014.12.31jichu.csv','丙氨酸氨基转移酶','basajzym.csv','a')
# getTestInfor('2015.1.1-2015.1.31jichu.csv','丙氨酸氨基转移酶','basajzym.csv','a')
# getTestInfor('2015.2.1-2015.2.28jichu.csv','丙氨酸氨基转移酶','basajzym.csv','a')
# getTestInfor('2015.3.1-2015.3.31jichu.csv','丙氨酸氨基转移酶','basajzym.csv','a')
# getTestInfor('2015.4.1-2015.4.31jichu.csv','丙氨酸氨基转移酶','basajzym.csv','a')
# getTestInfor('2015.5.1-2015.5.31jichu.csv','丙氨酸氨基转移酶','basajzym.csv','a')

def first12tempfun(filename,output,mode):
    csv_reader = csv.reader(open(filename))
    f = open(output, mode)
    x0 = []
    x1 = []
    x2 = []
    for i in csv_reader:
        x0.append(i[0])
        x1.append(i[1])
        x2.append(i[2])
    if len(x0)>=1:
        a=1
        count =1
        f.write(x0[a-1]+','+x1[a-1])
        while a<len(x0):
            if x0[a]!=x0[a-1]:
                if count<=12:
                    k=0
                    while k<(12-count):
                        f.write(','+x1[a-1])
                        k+=1
                f.write('\n'+x0[a]+','+x1[a])
                count =1
            else:
                if count <=12:
                    f.write(','+x1[a])
                    count+=1
            a+=1
        if a==len(x0) and count<=12:
            k = 0
            while k < (12 - count):
                f.write(',' + x1[a - 1])
                k += 1
            f.write('\n')
        else:
            f.write('\n')
    f.close()


first12tempfun('2014.6.1-2014.8.31tiwen.csv','tiwen.csv','w')
first12tempfun('2014.9.1-2014.10.31tiwen.csv','tiwen.csv','a')
first12tempfun('2014.11.1-2014.11.31tiwen.csv','tiwen.csv','a')
first12tempfun('2014.12.1-2014.12.31tiwen.csv','tiwen.csv','a')
first12tempfun('2015.01.01-2015.06.30tiwen.csv','tiwen.csv','a')






