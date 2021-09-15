
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.pylab import mpl
import matplotlib.pyplot as plt

df = pd.read_excel('all2.xlsx')
df = np.array(df)
x, y = np.split(df, (244,), axis=1)
x = x[:, 16:18]/100000
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

test_num = y_test.size
clf = svm.SVC(C=15, kernel='linear', decision_function_shape='ovo')
# clf = svm.SVC(C=3, kernel='rbf', gamma=0.1, decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())

w = clf.coef_[0]
k = -w[0] / w[1]
b = - (clf.intercept_[0]) / w[1]
print('k:',k)
print('b:',b)
print('wx:',w[0])
print('wy:',w[1])
print('intercept:',clf.intercept_[0])
# yy = a * xx - (clf.intercept_[0]) / w[1],w[0]*x+w[1]*y+intercept=0

print( clf.score(x_train, y_train) ) # 精度
# y_hat = clf.predict(x_train)
# show_accuracy(y_hat, y_train, '训练集')
print( clf.score(x_test, y_test) )
y_hat = clf.predict(x_test)
# show_accuracy(y_hat, y_test, '测试集')

print( 'decision_function(train):\n', clf.decision_function(x_train) )
print( 'decision_function(test):\n', clf.decision_function(x_test) )
print( '\npredict:\n', clf.predict(x_test) )

x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围

# x1_min, x1_max = 0, 1  # 第0列的范围
# x2_min, x2_max = 0, 1  # 第1列的范围

x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点
grid_test = np.stack((x1.flat, x2.flat), axis=1)
# 测试点
print ('grid_test = \n', grid_test)
grid_hat = clf.predict(grid_test)
# 预测分类值
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
s1 = plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolors='k', s=50, cmap=cm_dark)  # 样本
s2 = plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='x', s=50, cmap=cm_dark)  # 样本
# plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
plt.xlabel(u'Real Part of 16kHz', fontsize=13)
plt.ylabel(u'Real Part of 14kHz', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'SVM-binary classification', fontsize=15)

plt.legend((s1,s2),('train','test') ,loc = 'lower right')
# plt.grid()
plt.show()