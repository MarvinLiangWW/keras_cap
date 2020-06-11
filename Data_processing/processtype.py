# '''将所有的病例根据致病因分成三个文件储存
#     每个文件储存单一致病因的病例的编号'''
# import csv
# csv_reader1 = csv.reader(open('汇总表2015-10-5(fixed).csv', encoding='GBK'))
# '''GBK的原因：支（衣）原体肺炎在数据中有‘1’和‘衣原体’两种标识'''
# f = open('肺炎支（衣）原体肺炎.csv', 'w+')
# f2 = open('细菌性肺炎.csv', 'w+')
# f3 = open('病毒性肺炎.csv', 'w+')
# for i in csv_reader1:
#     x1=i[0]
#     x4=i[4]
#     x5=i[5]
#     x6=i[6]
#     x7=i[7]
#     x8=i[8]
#     if x4 != '' and x7 == '':
#         f.write(x1 + '\n')
#     if x5 != '' and x7 == '':
#         f3.write(x1 + '\n')
#     if x6 != '' and x8 == '':
#         f2.write(x1 + '\n')
# f.close()
# f2.close()
# f3.close()

import csv
def devidetwocol(filename,outputcol1,ouputcol2,mode):
    csv_reader = csv.reader(open(filename))
    f1=open(outputcol1,mode)
    f2 = open(ouputcol2,mode)
    for i in csv_reader:
        f1.write(i[0]+'\n')
        f2.write(i[1]+'\n')
    f1.close()
    f2.close()



devidetwocol('number-category.csv','number.csv','category.csv','w')



def abc():
    size = 10000
    np.random.seed(seed=10)
    X_seed = np.random.normal(0, 1, size)
    X0 = X_seed + np.random.normal(0, .1, size)
    X1 = X_seed + np.random.normal(0, .1, size)
    X2 = X_seed + np.random.normal(0, .1, size)
    X = np.array([X0, X1, X2]).T
    Y = X0 + X1 + X2

    rf = RandomForestRegressor(n_estimators=20, max_features=2)
    rf.fit(X, Y);
    print("Scores for X0, X1, X2:", map(lambda x: round(x, 3),rf.feature_importances_))


