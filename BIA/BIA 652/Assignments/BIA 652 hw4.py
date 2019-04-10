import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
import random
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

data = pd.read_excel('/Users/haodong/Desktop/Classification Data2.xlsx', head = 0)

# get dummy, 0 for Yes and 1 for No
data['Y'] = pd.get_dummies(data['Y'])
# print(data)

X, y = data.iloc[:, :-1], data.iloc[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(len(X_train), len(X_test), len(y_train), len(y_test))
#
# Q1
# class count
count_class_0, count_class_1 = y_train.Y.value_counts()
#
# # divide by class
df_class_0 = y_train[y_train['Y'] == 1]
df_class_1 = y_train[y_train['Y'] == 0]
#
# # find the count of different class
print('\nFollowing is for undersampling\n')
print('count of class 0:', len(df_class_0))
print('count of class 1:', len(df_class_1))
#
# # 1:364
# # 0:256
#
# # undersampling
df_class_0_under = df_class_0.sample(count_class_1)
# print(df_class_0_under)

df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)
print(df_test_under.Y.value_counts())

df_test_under.Y.value_counts().plot(kind='bar')
plt.show()
#
#
# # oversampling
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
print('\nFollowing is for oversampling\n')
# print(df_class_1_over)
#
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)
print(df_test_over.Y.value_counts())
df_test_over.Y.value_counts().plot(kind='bar')
plt.show()

# Q2
# Logistic regression
# print('\nLogistic regression')
# model_LR = LogisticRegression(solver='lbfgs', max_iter=5000)
# model_LR = model_LR.fit(X_train, np.ravel(y_train))
# a = model_LR.predict(X_test)
#
# # print(a)
# ac = model_LR.score(X_test, np.ravel(y_test))
# print(ac)
#
# #
# # # LDA
# #
# print('\nLinear Discriminant Analysis')
# model_LDA = LinearDiscriminantAnalysis()
# model_LDA = model_LDA.fit(X_train, np.ravel(y_train))
#
# b = model_LDA.predict(X_test)
# # print(b)
# LDA_score = model_LDA.score(X_test, np.ravel(y_test))
# print(LDA_score)
# #
# # # KNN
# #
# print('\nKNeighborsClassifier')
# for i in range(3, 11):
#     model = 'model_KNN_{}'.format(i)
#     model = KNeighborsClassifier(n_neighbors = i)
#     model.fit(X_train, np.ravel(y_train))
#     print('k = {}'.format(i), model.score(X_test, np.ravel(y_test)))
#
# # When k = 3, knn has best performance
# #
# # model_KNN = KNeighborsClassifier(n_neighbors = 3)
# # model_KNN = model_KNN.fit(X_train, np.ravel(y_train))
# #
# #
# # # NB
# print('\nNaive Bayes')
# model_NB = GaussianNB()
# model_NB = model_NB.fit(X_train, np.ravel(y_train))
# c = model_NB.predict(X_test)
# # print(c)
# NB_score = model_NB.score(X_test, np.ravel(y_test))
# print(NB_score)



# Q3
# print('\nEnsamble above four classifiers by using Majority vote')
# NB = GaussianNB()
# LR = LogisticRegression(solver='lbfgs', max_iter=5000)
# LDA = LinearDiscriminantAnalysis()
# KNN = KNeighborsClassifier()
#
# eclf = VotingClassifier(estimators=[('NB', NB), ('LR', LR), ('LDA', LDA), ('KNN', KNN)], voting='hard')
#
# print('\nTest model by using cross validation')
# for clf, label in zip([KNN, LDA, NB, LR, eclf], ['KNN', 'LDA', 'NB', 'LR', 'Ensemble']):
#     scores = cross_val_score(clf, X, np.ravel(y), cv=5, scoring='accuracy')
#     print('Accuracy: %0.4f (+/- %0.4f) [%s]' % (scores.mean(), scores.std(), label))
#
#
# print('\nTest model by split dataset to 75% training and 25% validation data')
# ss = eclf.fit(X_train, np.ravel(y_train))
# print(ss.score(X_test, np.ravel(y_test)))
