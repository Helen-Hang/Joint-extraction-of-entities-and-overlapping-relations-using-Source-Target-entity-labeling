# encoding=utf-8

from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
#支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']
#
opacity = 0.9

x = ['0.3', '0.4', '0.5', '0.6', '0.7']
y = [0.792, 0.824, 0.854, 0.885, 0.915]
y1=[0.897, 0.886, 0.869, 0.841, 0.812]
y2=[0.841, 0.854, 0.861, 0.862, 0.860]
plt.plot(x, y, marker='o', color='g', label=u'Precision',alpha=opacity)
plt.plot(x, y1, marker='*', color='b',label=u'Recall',alpha=opacity)
plt.plot(x, y2, marker='s', color='r',label=u'F1',alpha=opacity)
plt.legend(fontsize=15)  # 让图例生效
plt.xticks(range(5),fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(u"Relation threshold", fontsize=15) #X轴标签
plt.ylabel("Value of (P,R,F1)", fontsize=15) #Y轴标签
plt.savefig("./fig11-1.png")
plt.show()

x = ['0.3', '0.4', '0.5', '0.6', '0.7']
y = [0.785, 0.828, 0.856, 0.900, 0.911]
y1=[0.917, 0.896, 0.864, 0.820, 0.765]
y2=[0.846, 0.861, 0.860, 0.858, 0.832]
plt.plot(x, y, marker='o', color='g', label=u'Precision', alpha=opacity)
plt.plot(x, y1, marker='*', color='b',label=u'Recall', alpha=opacity)
plt.plot(x, y2, marker='s', color='r',label=u'F1', alpha=opacity)
plt.legend(fontsize=15)  # 让图例生效
plt.xticks(range(5),fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(u"Relation threshold", fontsize=15) #X轴标签
plt.ylabel("Value of (P,R,F1)", fontsize=15) #Y轴标签
plt.savefig("./fig11-2.png")
plt.show()
