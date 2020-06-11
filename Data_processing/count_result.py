import csv
csv_reader = csv.reader(open('2014.6.1-2014.8.31age.csv', encoding='utf-8'))
x9 = []
for i in csv_reader:
    x9.append([int(i[0]),float(i[1])])
print(x9)


