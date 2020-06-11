import csv


def combine_data(filename1, filename2, filename3):
    csv_reader_file1 = csv.reader(open(filename1))
    csv_reader_file21 = csv.reader(open(filename2))
    x1 = [i[0] for i in csv_reader_file21]
    print(x1)
    csv_reader_file22 = csv.reader(open(filename2))
    x2 = [i[1] for i in csv_reader_file22]
    print(x2)
    f = open(filename3, 'w+')

    for i in csv_reader_file1:
        k = 0
        while k < len(i):
            f.write(i[k]+',')
            k += 1
        j = 0
        while j < len(x1):
            if x1[j] == i[0]:
                f.write(x2[j])
                break
            j += 1
        f.write('\n')
    f.close()


combine_data('test1.csv', 'test2.csv', 'test3.csv')
