# journey gamma-top1

import torch
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np

#γ值与top-1图
x1 = np.arange(0, 1, 0.05)
y1 = [82.2, 83.4, 84.3, 86.2, 86.2, 85.7, 87.3, 87.2, 88.4, 90.5, 90.6, 90.7, 90.3, 89.8, 87.4, 82.4, 78.2, 76.9, 72.4,
      72.4]
y2 = [55.2, 60.4, 63.8, 65.2, 70.4, 71.2, 72.3, 73.2, 73.3, 74.5, 74.6, 74.7, 73.7, 73.8, 72.4, 71.4, 68.2, 57.3, 53.4,
      50.4]
# plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'r^-', color='r', linewidth=2, markersize=8)
plt.plot(x1, y2, '*-', color='b', linewidth=2, markersize=8)

plt.xlabel('γ', fontsize='10')  # 10px为默认值
plt.ylabel('top-1%')
plt.xlim(0, 1)
ax = plt.gca()
y_major_locator = MultipleLocator(5)
ax.yaxis.set_major_locator(y_major_locator)

plt.ylim(50, 100)
# plt.subplot(2, 1, 2)

# 线的标签
plt.legend(('CUHK-SYSU', 'RPW',), loc='upper right')
# plt.title('gama vs top-1')
# plt.figure(figsize=(500, 500))
plt.savefig("gamaVStop_1.pdf")
plt.show()




# 1 batch_size
batch_size = min(batch_size, len(dataset))

# 2 number of workers
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

