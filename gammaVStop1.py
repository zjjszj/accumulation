# journey gamma-top1
#γ值与top-1图
fig=plt.figure(figsize=(2.7, 2.2))
x1 = np.arange(0, 1, 0.05)
y1 = [82.2, 83.4, 84.3, 86.2, 86.2, 85.7, 87.3, 87.2, 88.4, 90.5, 90.6, 90.7, 90.3, 89.8, 87.4, 82.4, 78.2, 76.9, 72.4,
      72.4]
y2 = [55.2, 60.4, 63.8, 65.2, 70.4, 71.2, 72.3, 73.2, 73.3, 74.5, 74.6, 74.7, 73.7, 73.8, 72.4, 71.4, 68.2, 57.3, 53.4,
      50.4]
# plt.subplot(2, 1, 1)
plt.plot(x1, y1, color='black', linewidth=1)
plt.plot(x1, y2, '*-', color='black', linewidth=1, markersize=2)

plt.xlabel('γ', fontsize='7.5')  # 7.5px为六号字体
plt.ylabel('top-1%', fontsize='7.5')
plt.xlim(0, 1)
ax = plt.gca()
y_major_locator = MultipleLocator(5)
ax.yaxis.set_major_locator(y_major_locator)

plt.ylim(50, 100)
plt.tick_params(labelsize=7.5)   # s设置刻度字号
# plt.subplot(2, 1, 2)

# 线的标签
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 7.5,
}
plt.legend(('CUHK-SYSU', 'RPW',), loc='upper left',prop=font1, frameon=False)  # drop out frame.
plt.grid()
# plt.title('gama vs top-1')
plt.savefig("gamaVStop_1.jpg",dpi=300, bbox_inches = 'tight')  # set bbox_inches otherwise drop out xlabel and ylabel.
plt.show()