初步打算将每一个特征用三维的数据进行表示
第一维：0,1     0：表示无这类数据，1：表示有这类数据
第二维：具体的数值   若无则去所有有的数据的平均值
第三维：0,1     0：表示不在参考值的区间之间，1：表示在参考值的区间之间

值为未见或者0的参数：
红细胞形态,异常白细胞，异常红细胞，嗜碱细胞百分比

C反应蛋白,参考值是（0-8）mg/l

final_data.csv的数据组成
col0:   编号
col1:   年龄
col1:   血小板分布宽度的具体数值，没有则取所有值得平均值
col2:   血小板分布宽度的具体数值是否在参考值得区间之内，假设所有没有取值的参数都在参考值内
col3:     嗜碱细胞绝对值
col5:    红细胞总数
col7:   紅細胞分布宽度标准差  yes
col9:   平均红细胞体积
col11:   中性细胞绝对值    yes
col13:   平均红细胞血红蛋白浓度    yes
col15:  血小板总数   yes
col17:  血红蛋白   yes
col19:  淋巴细胞绝对值   yes
col21:  白细胞总数    yes
col23:  单核细胞百分比
col25:  单核细胞绝对值
col27:  淋巴细胞百分比   yes
col29:  嗜碱细胞百分比
col31:  嗜酸细胞百分比
col33:  血小板压积
col35:  大血小板比率  yes
col37:  红细胞压积
col39:  红细胞分布宽度变异系数   yes
col41:  中性细胞百分比   yes
col43:  平均红细胞血红蛋白含量
col45:  嗜酸细胞绝对值
col47:  血小板平均体积
col49:  C反应蛋白   yes

col51:  白蛋白
col53:  丙氨酸氨基转移酶
col55:  天门冬氨酸氨基转移酶



    # delete the file number_category.csv from line 1262 to 1577
    # what's more 10797 10801 10809 10810 10811 10820
    # data in 24551 is abnormal
    # col15 1756 from 6 to 600
    # 单位 col37 is different
    # baidanbao data is not accurate delete all data that less than 1
