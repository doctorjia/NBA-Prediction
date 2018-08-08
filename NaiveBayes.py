import numpy as np
import xlrd
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer


def read_excel():
    ExcelFile = xlrd.open_workbook(r'E:\SJTU\VE488\NB\new1617.xlsx')
    sheet = ExcelFile.sheet_by_index(0)
    return sheet


def read_val():
    ExcelFile = xlrd.open_workbook(r'E:\SJTU\VE488\NB\prediction.xlsx')
    sheet = ExcelFile.sheet_by_index(0)
    return sheet


fnn = buildNetwork(16, 8, 1)  # 第一个2是输入层的数据元（简单理解为有几个变量吧),第四个1是输出层的数据元（简单理解为因变量的个数）
sheet = read_excel()
train = np.array(sheet.col_values(0))  # 第一个自变量向量
train2 = np.array(sheet.col_values(1))  # 第二个自变量向量
train3 = np.array(sheet.col_values(2))
train4 = np.array(sheet.col_values(3))
train5 = np.array(sheet.col_values(4))
train6 = np.array(sheet.col_values(5))
train7 = np.array(sheet.col_values(6))
train8 = np.array(sheet.col_values(7))
train9 = np.array(sheet.col_values(8))
train10 = np.array(sheet.col_values(9))
train11 = np.array(sheet.col_values(10))
train12 = np.array(sheet.col_values(11))
train13 = np.array(sheet.col_values(12))
train14 = np.array(sheet.col_values(13))
train15 = np.array(sheet.col_values(14))
train16 = np.array(sheet.col_values(15))
label = np.array(sheet.col_values(16))  # 因变量
tmax = train.max()  # 归一化
tmin = train.min()
tmea = train.mean()
train = (train - tmin) * 1.0 / (tmax - tmin)
#train = (train - tmea) * 2.0 / (tmax - tmin)
tmin2 = train2.min()
tmax2 = train2.max()
tmea2 = train2.mean()
train2 = (train2 - tmin2) * 1.0 / (tmax2 - tmin2)
#train2 = (train2 - tmea2) * 2.0 / (tmax2 - tmin2)
tmin3 = train3.min()
tmax3 = train3.max()
tmea3 = train3.mean()
train3 = (train3 - tmin3) * 1.0 / (tmax3 - tmin3)
#train3 = (train3 - tmea3) * 2.0 / (tmax3 - tmin3)
tmin4 = train4.min()
tmax4 = train4.max()
tmea4 = train4.mean()
train4 = (train4 - tmin4) * 1.0 / (tmax4 - tmin4)
#train4 = (train4 - tmea4) * 2.0 / (tmax4 - tmin4)
tmin5 = train5.min()
tmax5 = train5.max()
tmea5 = train5.mean()
train5 = (train5 - tmin5) * 1.0 / (tmax5 - tmin5)
#train5 = (train5 - tmea5) * 2.0 / (tmax5 - tmin5)
tmin6 = train6.min()
tmax6 = train6.max()
tmea6 = train6.mean()
train6 = (train6 - tmin6) * 1.0 / (tmax6 - tmin6)
#train6 = (train6 - tmea6) * 2.0 / (tmax6 - tmin6)
tmin7 = train7.min()
tmax7 = train7.max()
tmea7 = train7.mean()
train7 = (train7 - tmin7) * 1.0 / (tmax7 - tmin7)
#train7 = (train7 - tmea7) * 2.0 / (tmax7 - tmin7)
tmin8 = train8.min()
tmax8 = train8.max()
tmea8 = train8.mean()
train8 = (train8 - tmin8) * 1.0 / (tmax8 - tmin8)
#train8 = (train8 - tmea8) * 2.0 / (tmax8 - tmin8)
tmin9 = train9.min()
tmax9 = train9.max()
train9 = (train9 - tmin9) * 1.0 / (tmax9 - tmin9)
tmin10 = train10.min()
tmax10 = train10.max()
train10 = (train10 - tmin10) * 1.0 / (tmax10 - tmin10)
tmin11 = train11.min()
tmax11 = train11.max()
train11 = (train11 - tmin11) * 1.0 / (tmax11 - tmin11)
tmin12 = train12.min()
tmax12 = train12.max()
train12 = (train12 - tmin12) * 1.0 / (tmax12 - tmin12)
tmin13 = train13.min()
tmax13 = train13.max()
train13 = (train13 - tmin13) * 1.0 / (tmax13 - tmin13)
tmin14 = train14.min()
tmax14 = train14.max()
train14 = (train14 - tmin14) * 1.0 / (tmax14 - tmin14)
tmin15 = train15.min()
tmax15 = train15.max()
train15 = (train15 - tmin15) * 1.0 / (tmax15 - tmin15)
tmin16 = train16.min()
tmax16 = train16.max()
train16 = (train16 - tmin16) * 1.0 / (tmax16 - tmin16)
lmax = label.max()
lmin = label.min()
lmea = label.mean()
label = (label - lmin) * 1.0 / (lmax - lmin)
#label = (label - lmea) * 2.0 / (lmax - lmin)
ds = SupervisedDataSet(16, 1)
#print(type(train[0]))
for i in range(len(train)):
    ds.addSample([train[i], train2[i], train3[i], train4[i], train5[i], train6[i], train7[i], train8[i], train9[i], train10[i], train11[i], train12[i], train13[i], train14[i], train15[i], train16[i]], [label[i]])
x = ds['input']
y = ds['target']
#print(ds)
trainer = BackpropTrainer(fnn, ds, verbose=True,learningrate=0.05)
trainer.trainEpochs(epochs=1000)  # 迭代次数

'''
train = np.hstack((train, np.array([0.36])))  # 添加自变量1 为3
print(train)
train2 = np.hstack((train2, np.array([6.75])))  # 添加自变量2 为4
train3 = np.hstack((train3, np.array([23.25])))  # 添加自变量3 为3
train4 = np.hstack((train4, np.array([17.775])))  # 添加自变量4 为4
train5 = np.hstack((train5, np.array([5.85])))  # 添加自变量5 为3
train6 = np.hstack((train6, np.array([77.55])))  # 添加自变量6 为4
label = np.hstack((label, np.array([2.739454094])))  # 添加因变量为403(4*100+3)
'''
sheet2 = read_val()
add = np.array(sheet2.col_values(0))
add2 = np.array(sheet2.col_values(1))
add3 = np.array(sheet2.col_values(2))
add4 = np.array(sheet2.col_values(3))
add5 = np.array(sheet2.col_values(4))
add6 = np.array(sheet2.col_values(5))
add7 = np.array(sheet2.col_values(6))
add8 = np.array(sheet2.col_values(7))
add9 = np.array(sheet2.col_values(8))
add10 = np.array(sheet2.col_values(9))
add11 = np.array(sheet2.col_values(10))
add12 = np.array(sheet2.col_values(11))
add13 = np.array(sheet2.col_values(12))
add14 = np.array(sheet2.col_values(13))
add15 = np.array(sheet2.col_values(14))
add16 = np.array(sheet2.col_values(15))
lab = np.array(sheet2.col_values(16))
add = (add - tmin) * 1.0 / (tmax - tmin)
add2 = (add2 - tmin2) * 1.0 / (tmax2 - tmin2)
add3 = (add3 - tmin3) * 1.0 / (tmax3 - tmin3)
add4 = (add4 - tmin4) * 1.0 / (tmax4 - tmin4)
add5 = (add5 - tmin5) * 1.0 / (tmax5 - tmin5)
add6 = (add6 - tmin6) * 1.0 / (tmax6 - tmin6)
add7 = (add7 - tmin7) * 1.0 / (tmax7 - tmin7)
add8 = (add8 - tmin8) * 1.0 / (tmax8 - tmin8)
add9 = (add9 - tmin9) * 1.0 / (tmax9 - tmin9)
add10 = (add10 - tmin10) * 1.0 / (tmax10 - tmin10)
add11 = (add11 - tmin11) * 1.0 / (tmax11 - tmin11)
add12 = (add12 - tmin12) * 1.0 / (tmax12 - tmin12)
add13 = (add13 - tmin13) * 1.0 / (tmax13 - tmin13)
add14 = (add14 - tmin14) * 1.0 / (tmax14 - tmin14)
add15 = (add15 - tmin15) * 1.0 / (tmax15 - tmin15)
add16 = (add16 - tmin16) * 1.0 / (tmax16 - tmin16)
lab = (lab - lmin) * 1.0 / (lmax - lmin)
#add = (add - tmea) * 2.0 / (tmax - tmin)
#add2 = (add2 - tmea2) * 2.0 / (tmax2 - tmin2)
#add3 = (add3 - tmea3) * 2.0 / (tmax3 - tmin3)
#add4 = (add4 - tmea4) * 2.0 / (tmax4 - tmin4)
#add5 = (add5 - tmea5) * 2.0 / (tmax5 - tmin5)
#add6 = (add6 - tmea6) * 2.0 / (tmax6 - tmin6)
#add7 = (add7 - tmea7) * 2.0 / (tmax7 - tmin7)
#add8 = (add8 - tmea8) * 2.0 / (tmax8 - tmin8)
#lab = (lab - lmea) * 2.0 / (lmax - lmin)
#print(add)
#print(type(add[0]))

for i in range(len(add)):
    '''
    a = float(add[i])
    a2 = float(add2[i])
    a3 = float(add3[i])
    a4 = float(add4[i])
    a5 = float(add5[i])
    a6 = float(add6[i])
    a7 = float(add7[i])
    a8 = float(add8[i])
    l = float(lab[i])
    '''
    out = SupervisedDataSet(16, 1)
    out.addSample([add[i], add2[i], add3[i], add4[i], add5[i], add6[i], add7[i], add8[i], add9[i], add10[i], add11[i], add12[i], add13[i], add14[i], add15[i], add16[i]], [lab[i]])
    out = fnn.activateOnDataset(out)
    out = out * (lmax - lmin) + lmin  # 求得原始数据
    #out = out * (lmax - lmin) / 2.0 + lmea  # 求得原始数据
    print(str(i) + " is " + str(out))

