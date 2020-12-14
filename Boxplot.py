import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
# 添加成绩表
plt.style.use("ggplot")
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
# 新建一个空的DataFrame
df = pd.DataFrame()
#NYT
df["CopyRE"] = [0.587,0.595,0.589,0.581,0.592,0.595,0.585,0.591,0.588,0.592]
df["CopyMTL"] = [0.663,0.667,0.668,0.654,0.667,0.673,0.659,0.653,0.669,0.677]
df[ r"$CopyR_{RL}$"] = [0.708,0.71,0.713,0.706,0.702,0.703,0.706,0.704,0.704,0.708]
df["BERT-JEORE"] = [0.86243,0.86496,0.86434,0.8646,0.86318,0.86752,0.86538,0.86807,0.86393,0.86422]

sns.set_style("white")
sns.set_style("ticks")
sns.boxplot(data=df,showfliers=False)
y_tick = np.linspace(0.55,0.90,8)
plt.xticks(fontsize=15)
plt.yticks(y_tick, fontsize=15)
plt.savefig("./fig10-1.png")
plt.show()

#WebNLG
df["CopyRE"] = [0.371,0.36,0.359,0.366,0.366,0.351,0.36,0.362,0.356,0.362]
df["CopyMTL"] = [0.589,0.573,0.58,0.538,0.562,0.531,0.526,0.534,0.548,0.554]
df[ r"$CopyR_{RL}$"] = [0.586,0.579,0.583,0.579,0.581,0.584,0.585,0.586,0.582,0.587]
df["BERT-JEORE"] = [0.84622,0.84996,0.85203,0.84548,0.84868,0.84946,0.8446,0.85217,0.84334,0.84825]

sns.set_style("white")
sns.set_style("ticks")
sns.boxplot(data=df,showfliers=False)
y_tick = np.linspace(0.35,0.90,12)
plt.xticks(fontsize=15)
plt.yticks(y_tick,fontsize=15)
plt.savefig("./fig10-2.png")
plt.show()

#figure+subfigure
# import matplotlib.pyplot as plt
# import pandas as pd
#
# if __name__ == "__main__":
#     datafile = '/home/htt/Desktop/NYT-Multi/Boxplot-NYT.xlsx'
#     data = pd.read_excel(datafile)
#     length = len(data.columns)
#     # 统计每列值的max-min值
#     delta = 0
#     for i in range(length):
#         delta = max(delta, max(data[data.columns[i]]
#                                )-min(data[data.columns[i]]))/home/htt/Desktop/Webnlg/raw_data/webnlg1/input/train.json
#     delta *= 1.1
#     # 计算每列值的y基底值
#     y_bottom = []
#     for i in range(length):
#         ave = (max(data[data.columns[i]]) + min(data[data.columns[i]])) / 2
#         y_bottom.append(ave - delta / 2)
#
#     fig = plt.figure()
#     main = plt.subplot2grid((length+1, length), (0, 0), rowspan=length-1, colspan=length)
#     main.boxplot(data, labels=data.columns)
#
#     for i in range(length):
#         sub = plt.subplot2grid((length+1, length), (length-1, i), rowspan=2)
#         sub.set_ylim(y_bottom[i], y_bottom[i]+delta) # 设置y轴高度
#         sub.boxplot(data[data.columns[i]], labels=[data.columns[i]])
#
#     plt.tight_layout()
#     plt.show()

