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