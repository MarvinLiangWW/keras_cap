import csv

csv_readerage = csv.reader(open("2014.6.1-2014.8.31age.csv"))
csv_readeravgtemp1 = csv.reader(open("2014.6.1-2014.8.31avgtemp.csv"))
x1 = [i[0]for i in csv_readeravgtemp1]
csv_readeravgtemp2 = csv.reader(open("2014.6.1-2014.8.31avgtemp.csv"))
x2 = [i[1]for i in csv_readeravgtemp2]
f=open('finalData.csv','w+')

for i in csv_readerage:
    f.write(i[0]+','+i[1]+',')
    j=0
    while j < len(x1):
        if x1[j]==i[0]:
            f.write(x2[j])
            break
        j+=1
    f.write('\n')
f.close()


def combinedata(filename1, filename2, filename3):
    csv_readerfile1 = csv.reader(open(filename1))
    csv_readerfile21 = csv.reader(open(filename2))
    x1 = [i[0] for i in csv_readerfile21]
    print(x1)
    csv_readerfile22 = csv.reader(open(filename2))
    x2 = [i[1] for i in csv_readerfile22]
    print(x2)
    f = open(filename3, 'w+')

    for i in csv_readerfile1:
        k=0
        while k<len(i):
            f.write(i[k]+',')
            k+=1
        j = 0
        while j < len(x1):
            if x1[j] == i[0]:
                f.write(x2[j])
                break
            j += 1
        f.write('\n')
    f.close()


combinedata('test1.csv','test2.csv','test3.csv')