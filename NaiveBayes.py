import numpy as np
import xlrd
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer


def read_excel():
    ExcelFile = xlrd.open_workbook(r'E:\SJTU\VE488\NB\testnb.xlsx')
    sheet = ExcelFile.sheet_by_index(0)
    return sheet


def read_val():
    ExcelFile = xlrd.open_workbook(r'E:\SJTU\VE488\NB\validation.xlsx')
    sheet = ExcelFile.sheet_by_index(0)
    return sheet


fnn = buildNetwork(8, 4, 1)  # 第一个2是输入层的数据元（简单理解为有几个变量吧),第四个1是输出层的数据元（简单理解为因变量的个数）
sheet = read_excel()
train = np.array(sheet.col_values(0))  # 第一个自变量向量
train2 = np.array(sheet.col_values(1))  # 第二个自变量向量
train3 = np.array(sheet.col_values(2))
train4 = np.array(sheet.col_values(3))
train5 = np.array(sheet.col_values(4))
train6 = np.array(sheet.col_values(5))
train7 = np.array(sheet.col_values(6))
train8 = np.array(sheet.col_values(7))
label = np.array(sheet.col_values(8))  # 因变量
tmax = train.max()  # 归一化
tmin = train.min()
train = (train - tmin) * 1.0 / (tmax - tmin)
tmin2 = train2.min()
tmax2 = train2.max()
train2 = (train2 - tmin2) * 1.0 / (tmax2 - tmin2)
tmin3 = train3.min()
tmax3 = train3.max()
train3 = (train3 - tmin3) * 1.0 / (tmax3 - tmin3)
tmin4 = train4.min()
tmax4 = train4.max()
train4 = (train4 - tmin4) * 1.0 / (tmax4 - tmin4)
tmin5 = train5.min()
tmax5 = train5.max()
train5 = (train5 - tmin5) * 1.0 / (tmax5 - tmin5)
tmin6 = train6.min()
tmax6 = train6.max()
train6 = (train6 - tmin6) * 1.0 / (tmax6 - tmin6)
tmin7 = train7.min()
tmax7 = train7.max()
train7 = (train7 - tmin7) * 1.0 / (tmax7 - tmin7)
tmin8 = train8.min()
tmax8 = train8.max()
train8 = (train8 - tmin8) * 1.0 / (tmax8 - tmin8)
lmax = label.max()
lmin = label.min()
label = (label - lmin) * 1.0 / (lmax - lmin)
ds = SupervisedDataSet(8, 1)
print(type(train[0]))
for i in range(len(train)):
    ds.addSample([train[i], train2[i], train3[i], train4[i], train5[i], train6[i], train7[i], train8[i]], [label[i]])
x = ds['input']
y = ds['target']
print(ds)
trainer = BackpropTrainer(fnn, ds, verbose=True, learningrate=0.01)
trainer.trainEpochs(epochs=500)  # 迭代次数

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
lab = np.array(sheet2.col_values(8))
add = (add - tmin) * 1.0 / (tmax - tmin)
add2 = (add2 - tmin2) * 1.0 / (tmax2 - tmin2)
add3 = (add3 - tmin3) * 1.0 / (tmax3 - tmin3)
add4 = (add4 - tmin4) * 1.0 / (tmax4 - tmin4)
add5 = (add5 - tmin5) * 1.0 / (tmax5 - tmin5)
add6 = (add6 - tmin6) * 1.0 / (tmax6 - tmin6)
add7 = (add7 - tmin7) * 1.0 / (tmax7 - tmin7)
add8 = (add8 - tmin8) * 1.0 / (tmax8 - tmin8)
lab = (lab - lmin) * 1.0 / (lmax - lmin)
print(add)
print(type(add[0]))

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
    out = SupervisedDataSet(8, 1)
    out.addSample([add[i], add2[i], add3[i], add4[i], add5[i], add6[i], add7[i], add8[i]], [lab[i]])
    out = fnn.activateOnDataset(out)
    out = out * (lmax - lmin) + lmin  # 求得原始数据
    print(str(i) + " is " + str(out))

