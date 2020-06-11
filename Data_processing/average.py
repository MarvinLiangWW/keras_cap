import csv
from datetime import datetime, timedelta
# number
csv_reader1 = csv.reader(open('2014.6.1-2014.8.31temperature.csv', encoding='utf-8'))
x1 = [i[0]for i in csv_reader1]
# temperature
csv_reader2 = csv.reader(open('2014.6.1-2014.8.31temperature.csv', encoding='utf-8'))
x2 = [i[1]for i in csv_reader2]
# timestamp
csv_reader3 = csv.reader(open('2014.6.1-2014.8.31temperature.csv', encoding='utf-8'))
x3 = [i[2]for i in csv_reader3]

f = open('2014.6.1-2014.8.31avgtemp.csv', 'w+')
# calculate the average temperature of the first 24 hours in the hospital
count = 1
avg = 0.0
i = 0
time = x3[0]
while i < len(x1):
    j = i + 1
    avg += float(x2[i])
    if j != len(x1):
        if x1[i] == x1[j]:
            count += 1
        else:
            f.write(x1[i]+',')
            f.write(str(avg/count)+'\n')
            avg = 0.0
            count = 1
    else:
        f.write(x1[i]+',')
        f.write(str(avg / count)+'\n')
    i += 1

f.close()