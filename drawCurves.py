import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
file_path = './logs/2020-05-23/r06_manual/2020-05-23_10-47_loss.txt'
loss = []
with open(file_path, 'r') as f:
    for line in f:
        ls = line.strip().split()
        temp = ls[2].split(':')[1]
        loss.append(temp)
length = len(loss)
x = np.arange(length)
y = [float(y) for y in loss]
# data1_loss = np.loadtxt("./experiment/2019-11-13_17-17/logs/2019-11-13_17-17_loss.txt")
# x = data1_loss[:, 0]
# y = data1_loss[:, 1]


fig = plt.figure(figsize=(7, 5))       #figsize是图片的大小`
#ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`

pl.plot(x, y, 'g-', label=u'Denoise')
# ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
#p2 = pl.plot(x, y,'r-', label = u'Point_Net')
pl.legend()
#显示图例
pl.xlabel(u'epoch')
pl.ylabel(u'loss')
plt.title('loss for model in training')
plt.show()
