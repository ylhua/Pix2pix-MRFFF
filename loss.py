from matplotlib.font_manager import FontManager
fm = FontManager()
mat_fonts = set(f.name for f in fm.ttflist)
print(mat_fonts)


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv
from scipy import interpolate
from scipy.interpolate import spline
'''读取csv文件'''
from scipy.signal import convolve
# if z is 1d array, m is the window shape(moving average num) AR PL UMing CN


def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        m = int(row[1])
        n = float(row[2])
        y.append((float(row[2])))
        x.append((int(row[1])))
    return x, y


mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

plt.figure()
x2, y2 = readcsv("./smooth_unet_d_real.csv")
# xnew = np.arange(400, 60000, 100)
#
# # 实现函数
# func = interpolate.interp1d(x2, y2, kind='quadratic')
#
# # 利用xnew和func函数生成ynew,xnew数量等于ynew数量
# ynew = func(xnew)
# ynew = convolve(ynew, np.ones(1))

plt.plot(x2, y2, color='red', label='D_real')
# plt.plot(x2, y2, '.', color='red')

x, y = readcsv("./smooth_unet_d_fake.csv")
xnew = np.linspace(min(x), max(x), 3000)  # 300 represents number of points to make between T.min and T.max
#
power_smooth = spline(x, y, xnew)/100
# # power_smooth = power_smooth

plt.plot(x, y, 'g', label='D_fake')
# #
# x1, y1 = readcsv("scalars2.csv")
# plt.plot(x1, y1, color='black', label='Without DW and PW')
#
# x4, y4 = readcsv("./trans.csv")
# plt.plot(x4, y4, color='blue', label='Trans_loss')

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.ylim(0, 1)
plt.xlim(0, 60000)
plt.xlabel('step', fontsize=10, labelpad=6, x=1.02)
plt.ylabel('loss', rotation='horizontal', fontsize=10, y=1.02)
plt.legend(fontsize=10)
plt.savefig('./unet_D_loss.jpg')
plt.show()