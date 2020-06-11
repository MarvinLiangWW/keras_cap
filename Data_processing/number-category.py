import csv
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

getnumber_category()

