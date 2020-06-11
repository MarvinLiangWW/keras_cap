import csv
from datetime import datetime, timedelta

csv_reader = csv.reader(open('huizongbasic.csv', encoding='GBK'))
f = open('huizongnumberage.csv', 'w+')
for i in csv_reader:
    birth_date = datetime.strptime(i[1], "%Y/%m/%d")
    start_date = datetime.strptime(i[2], "%Y/%m/%d %H:%M")
    time = (start_date - birth_date).days
    f.write(i[0]+','),
    temp = '%.2f'% (time/365)
    f.write(str(temp)+'\n')
f.close()







