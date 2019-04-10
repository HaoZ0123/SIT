import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

data = pd.read_excel("/Users/haodong/Desktop/bbb.xlsx")

x, y = data.iloc[:,:-1], data.iloc[:,-1:]

X_train, X_test, y_train, y_test = train_test_split(x, y,\
                test_size = 0.1, random_state = 0)

training, validation = X_train, X_test



# print(type(X_train))
# print(type(X_test))
# print(type(y_test))
# print(type(y_train))
# print(type(data))


# print('\nThe size of training set is: ', len(X_train))
# print('\nThe size of validation set is: ', len(X_test))




regressor = LinearRegression()

a = regressor.fit(X_train, y_train)

# y_pred = a.predict(X_test)
# plt.scatter(np.arange(len(y_test)), y_test, color = 'red', label = 'y_test')
# plt.scatter(np.arange(len(y_test)), y_pred, color = 'blue', label = 'y_pred')
# plt.legend(loc = 2)
# plt.show()

# print('\nThe R-squared value of the model is ', a.score(X_test, y_test))


#
# model1 = smf.ols(formula = 'y ~ x1', data = data).fit()
# summary1 = model1.summary()
# print(summary1)
# #
# model2 = smf.ols(formula = 'y ~ x2', data = data).fit()
# summary2 = model2.summary()
# print(summary2)
#
# model3 = smf.ols(formula = 'y ~ x3', data = data).fit()
# summary3 = model3.summary()
# print(summary3)
#
# model4 = smf.ols(formula = 'y ~ x4', data = data).fit()
# summary4 = model4.summary()
# print(summary4)
#
# model5 = smf.ols(formula = 'y ~ x5', data = data).fit()
# summary5 = model5.summary()
# print(summary5)
#
# model6 = smf.ols(formula = 'y ~ x6', data = data).fit()
# summary6 = model6.summary()
# print(summary6)
#
# model7 = smf.ols(formula = 'y ~ x7', data = data).fit()
# summary7 = model7.summary()
# print(summary7)
#
# model8 = smf.ols(formula = 'y ~ x8', data = data).fit()
# summary8 = model8.summary()
# print(summary8)
#
# model9 = smf.ols(formula = 'y ~ x9', data = data).fit()
# summary9 = model9.summary()
# print(summary9)
#
# model10 = smf.ols(formula = 'y ~ x10', data = data).fit()
# summary10 = model10.summary()
# print(summary10)

# mod001 = smf.ols(formula = 'y ~ x1 + x4', data = data).fit()
# sum001 = mod001.summary()
# print(sum001)
#
# mod002 = smf.ols(formula = 'y ~ x1 + x4 + x1:x4', data = data).fit()
# sum002 = mod002.summary()
# print(sum002)

# mod003 = smf.ols(formula = 'y ~ x1 + x4 + x3', data = data).fit()
# sum003 = mod003.summary()
# print(sum003)
#
# mod004 = smf.ols(formula = 'y ~ x1 + x4 + x1:x4 + x3', data = data).fit()
# sum004 = mod004.summary()
# print(sum004)
#
# mod005 = smf.ols(formula = 'y ~ x1 + x4 + x1:x4 + x3 + x1:x3+ x3:x4', data = data).fit()
# sum005 = mod005.summary()
# print(sum005)

# mod006 = smf.ols(formula = 'y ~ x1 + x4 + x1:x4 + x3 + x1:x3+ x3:x4 + x5 + x1:x5 + x3:x5 + x4:x5', data = data).fit()
# sum006 = mod006.summary()
# print(sum006)


# mod007 = smf.ols(formula = 'y ~ x1 + x4 + x1:x4 + x3 + x1:x3+ x3:x4 + x5 + x1:x5 + x3:x5 + x4:x5'
#                            '+ x2 + x1:x2 + x3:x2 + x4:x2 + x5:x2', data = data).fit()
# sum007 = mod007.summary()
# print(sum007)

# mod008 = smf.ols(formula = 'y ~ x1 + x4 + x1:x4 + x3 + x1:x3+ x3:x4 + x5 + x1:x5 + x3:x5 + x4:x5'
#                            '+ x2 + x1:x2 + x3:x2 + x4:x2 + x5:x2 + x7 + x7:x1 + x4:x7 + x5:x7 + x2:x7'
#                            , data = data).fit()
# sum008 = mod008.summary()
# print(sum008)

# mod009 = smf.ols(formula = 'y ~ x1 + x4 + x1:x4 + x3 + x1:x3+ x3:x4 + x5 + x1:x5 + x3:x5 + x4:x5'
#                            '+ x2 + x1:x2 + x3:x2 + x4:x2 + x5:x2 + x7 + x7:x1 + x4:x7 + x5:x7 + x2:x7'
#                            '+ x6 + x10 + x5:x10', data = data).fit()
# sum009 = mod009.summary()
# print(sum009)

# mod010 = smf.ols(formula = 'y ~ x1 + x4 + x1:x4 + x3 + x1:x3+ x3:x4 + x5 + x1:x5 + x3:x5 + x4:x5'
#                            '+ x2 + x1:x2 + x3:x2 + x4:x2 + x5:x2 + x7 + x7:x1 + x4:x7 + x5:x7 + x2:x7'
#                            '+ x6 + x10 + x5:x10 + x9 + x1:x9', data = data).fit()
# sum010 = mod010.summary()
# print(sum010)

# mod011 = smf.ols(formula = 'y ~ x1 + x4 + x3 + x1:x3+ x3:x4 + x1:x5'
#                            '+ x2 + x1:x2 + x3:x2 + x4:x2 + x5:x2 + x7 + x7:x1 + x4:x7 + x5:x7 + x2:x7'
#                            '+ x6 + x10 + x5:x10 + x9 + x1:x9', data = data).fit()
# sum011 = mod011.summary()
# print(sum011)
#
# mod012 = smf.ols(formula = 'y ~ x1 + x4 + x3 + x5 + x2 + x7 + x6 + x10 + x9', data = data).fit()
# sum012 = mod012.summary()
# print(sum012)

# mod013 = smf.ols(formula = 'y ~ x1 + x1:x2 + x1:x3 + x1:x4 + x1:x5 + x1:x6 + x1:x7 + x1:x8 + x1:x9 + x1:x10 + '
#                                'x2 + x2:x3 + x2:x4 + x2:x5 + x2:x6 + x2:x7 + x2:x8 + x2:x9 + x2:x10 + '
#                                'x3 + x3:x4 + x3:x5 + x3:x6 + x3:x7 + x3:x8 + x3:x9 + x3:x10 + '
#                                'x4 + x4:x5 + x4:x6 + x4:x7 + x4:x8 + x4:x9 + x4:x10 + '
#                                'x5 + x5:x6 + x5:x7 + x5:x8 + x5:x9 + x5:x10 + '
#                                'x6 + x6:x7 + x6:x8 + x6:x9 + x6:x10 + '
#                                'x7 + x7:x8 + x7:x9 + x7:x10 + '
#                                'x8 + x8:x9 + x8:x10 + '
#                                'x9 + x9:x10 + '
#                                'x10', data = data).fit()
# sum013 = mod013.summary()
# print(sum013)

# mod014 = smf.ols(formula = 'y ~ x1 + x1:x2 + x1:x3 + x1:x4 + x1:x5 + x1:x6 + x1:x7 + x1:x9 + x1:x10 + '
#                                'x2 + x2:x3 + x2:x4 + x2:x5 + x2:x7 + x2:x8 + x2:x10 + '
#                                'x3 + x3:x4 + x3:x5 + x3:x7 + x3:x8 + x3:x9 + '
#                                'x4 + x4:x5 + x4:x7 + x4:x8 + x4:x9 + x4:x10 + '
#                                'x5 + x5:x7 + x5:x9 + x5:x10 + '
#                                'x6 + x6:x10 + '
#                                'x7 + '
#                                'x8 + x8:x9 + '
#                                'x9 ', data = data).fit()
# sum014 = mod014.summary()
# print(sum014)


# mod013 = smf.ols(formula = 'y ~ x1 + x2 + x3 + x4 + x6 + x7', data = data).fit()
# sum013= mod013.summary()
# print(sum013)


# data1 = data.copy()
# data1['x3'] = np.log(data['x3'])

# mod014 = smf.ols(formula = 'y ~ x1 + x2 + x3 + x4 + x6 ', data = data1).fit()
# sum014= mod014.summary()
# print(sum014)

# mod014 = smf.ols(formula = 'y ~ x1 + x1:x2 + x1:x3 + x1:x4 + x1:x6 + x1:x7 + x1:x9 + x1:x10 + '
#                                'x2:x3 + x2:x7 + x2:x10 + '
#                                'x3 + x3:x4 + '
#                                'x4 + x4:x6 + x4:x7 + x4:x10 + '
#                                'x5:x7 + x5:x10 + '
#                                'x6:x7 + '
#                                'x7 ', data = data1).fit()
# sum014= mod014.summary()
# print(sum014)


data1 = data.copy()
data1['x3'] = np.log(data['x3'])
#
# mod013 = smf.ols(formula = 'y ~ x1 + x1:x2 + x1:x3 + x1:x4 + x1:x5 + x1:x6 + x1:x7 + x1:x8 + x1:x9 + x1:x10 + '
#                                'x2 + x2:x3 + x2:x5 + x2:x6 + x2:x7 + x2:x8 + x2:x9 + x2:x10 + '
#                                'x3 + x3:x4 + x3:x5 + x3:x8 + x3:x9 + '
#                                'x4 + x4:x6 + x4:x7 + x4:x8 + x4:x9 + '
#                                'x5 + x5:x7 + x5:x8 + x5:x10 + '
#                                'x6 + x6:x7 + x6:x10 + '
#                                'x7:x8 + x7:x10 + '
#                                'x8 + '
#                                'x10', data = data1).fit()
# sum013 = mod013.summary()
# print(sum013)
#

mod014 = smf.ols(formula = 'y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x1:x2 + x2:x6', data = data1).fit()
sum014= mod014.summary()
print(sum014)