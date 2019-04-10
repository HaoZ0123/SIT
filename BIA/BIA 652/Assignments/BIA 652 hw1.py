import numpy as np
import pandas as pd
import xlrd as xl
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.stats.stats import pearsonr
import seaborn as sb
from pylab import rcParams
from statsmodels.formula.api import ols
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm


data1 = pd.read_excel('/Users/haodong/Desktop/aaa.xlsx')

# q1

# x = data1['x1']
# y = data1['y']
#
# # plt.plot(x,y)
# plt.scatter(x,y)
# plt.xlabel("x1")
# plt.ylabel("y")
# plt.title("x1 vs y")
# plt.show()

# # q2
# pd.set_option('display.max_columns', None)
# df = pd.DataFrame(data1, columns= ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'])
# print(df)
# print(df.corr(method='kendall'))
# # sb.pairplot(data1)
# # plt.show()

# q3
# sb.boxplot(x = data1['y'])
# plt.show()

# data = pd.DataFrame(data1, columns= ['y'])
# # print(data)
#
# quartile_1, quartile_3 = np.percentile(data, [25, 75])
# iqr = quartile_3 - quartile_1
# lower_bound = quartile_1 - (iqr * 1.5)
# upper_bound = quartile_3 + (iqr * 1.5)
# outlier = np.where((data > upper_bound) | (data < lower_bound))
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# print(outlier)

# q4
# X = data1['ln(x3)']
# Y = data1['y']
#
# model = ols('Y ~ X', data1).fit()
# Y_pred = model.fittedvalues
# print(model.summary())
# print(model.params)
#
# #
# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # ax.scatter(X, Y, c = 'b')
# # ax.plot(X, Y_pred, c = 'r')
# # plt.show()
#
# aov_table = sm.stats.anova_lm(model, typ=2)
# print(aov_table)