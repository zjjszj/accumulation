import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import matplotlib.ticker as mticker


fig=plt.figure(figsize=(2.7, 2.2))
x1 = [ 1, 3, 5, 10, 15, 30]
y1 = [2.5, 2.9, 3.2, 3.3, 3.2, 5.2]
y2 = [ 1.1, 2.0, 2.1, 2.3, 2.2, 3.2]
y3 = [ 1.1, 1.1, 1.2, 1.3, 1.2, 1.5]
# plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'g.--', linewidth=1, markersize=3)
plt.plot(x1, y2, 'b<-.', linewidth=1, markersize=3)
plt.plot(x1, y3, 'm>-',  linewidth=1, markersize=3)
plt.xticks(x1)
# plt.xlabel('γ', fontsize='7.5')  # 7.5px为六号字体
plt.ylabel("delay time(s)", fontsize='7.5')
plt.xlabel("number of threads(thousand)", fontsize='7.5')
# plt.xlim(0, 15000)
ax = plt.gca()
y_major_locator = MultipleLocator(1)  # 设置y轴置刻度
# ax.yaxis.set_major_locator(y_major_locator)

# ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('dddd'))
# xticks=x1
# ax.set_xticks(xticks)


plt.ylim(0, 7)
plt.tick_params(labelsize=7.5)   # 设置刻度字号
# plt.subplot(2, 1, 2)

# 线的标签
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 7.5,
}
plt.legend(('QoS2', 'QoS1','QoS0'), loc='upper left',prop=font1)  # drop out frame.
# plt.grid()
# plt.title('gama vs top-1')
plt.savefig("gamaVStop_1.jpg",dpi=300, bbox_inches = 'tight')  # set bbox_inches otherwise drop out xlabel and ylabel.
plt.show()

# 画散点图
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


# # 坐标范围
plt.xlim(10, 25)
plt.ylim(10, 25)

# 生成数据集
x, y = make_blobs(n_samples=200, n_features=2, centers=[[12, 12], [13, 13],[14,14] ,[15, 15],[16,16],[17,17], [18, 18],[19,19] ,[20, 20],[21,21],[22,22],[23,23]],
                  cluster_std=[0.4, 1, 0.6, 0.7, 0.9, 0.8, 0.6,0.8,0.9, 0.4, 0.6, 0.5])
# 生成数据散点图
plt.scatter(x[:, 0], x[:, 1], marker='o')
# #直线
x=np.linspace(10, 24, 15)
y=x
plt.plot(x, y, '-r', label='y=2x+1',color='black')

#设置中文
from matplotlib import font_manager#导入字体管理模块
my_font = font_manager.FontProperties(fname="C:/WINDOWS/Fonts/msyh.ttc")

plt.xlabel('模拟值/℃', fontproperties = my_font,fontsize = 12)
plt.ylabel('实际值/℃', fontproperties = my_font,fontsize = 12)
plt.savefig('ddd.png')
plt.show()

#图
# 0 125 175 200 280, 400
x=[34, 36, 39,45, 56,78,105, 122, 156, 180,218, 248]
y=[33, 45, 39,49, 60,83,101, 124, 165, 177,224, 259]

#                   记号形状       颜色           点的大小    设置标签
plt.scatter(x, y, marker = 'o', color = 'black', s = 60)

# 线
x1=np.linspace(1, 300, 300)
y1 = x1
y2=(20/19)*x1
y3=(19/20)*x1
plt.plot(x1, y1, '-r', label='y=2x+1',color='black')


#设置中文
from matplotlib import font_manager#导入字体管理模块
my_font = font_manager.FontProperties(fname="C:/WINDOWS/Fonts/msyh.ttc")

plt.xlabel('实际天数/d', fontproperties = my_font,fontsize = 12)
plt.ylabel('模拟天数/d', fontproperties = my_font,fontsize = 12)

plt.savefig('ddd.png')
plt.show()