import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics


data = pd.read_excel('/Users/haodong/Desktop/hw3.xlsx', header=0)
data = data.dropna()
X, Y = data.iloc[:,:-1], data.iloc[:,-1:]

X['x1'] = pd.get_dummies(X['x1'])
X['x9'] = pd.get_dummies(X['x9'])
X['x10'] = pd.get_dummies(X['x10'])
X['x12'] = pd.get_dummies(X['x12'])
Y['Y'] = pd.get_dummies(Y['Y'])

classMap4 = {'u':0, 'y':1, 'l':2}
X['x4'] = X['x4'].map(classMap4)

classMap13 = {'g':0, 's':1, 'p':2}
X['x13'] = X['x13'].map(classMap13)

X_train, X_test, y_train, y_test = train_test_split(X, Y,\
                test_size = 0.2, random_state = 0)
print(X)

# # binary
# print('x1', X['x1'].unique())
#
# print('x2', X['x2'].unique())
# print('x3', X['x3'].unique())
#
# # recode u y l
# print('x4', X['x4'].unique())
# print('x8', X['x8'].unique())
#
# # binary
# print('x9', X['x9'].unique())
# # binary
# print('x10', X['x10'].unique())
#
# print('x11', X['x11'].unique())
#
# # binary
# print('x12', X['x12'].unique())
#
# # recode 3 g s p
# print('x13', X['x13'].unique())
# print('x14', X['x14'].unique())
# print('x15', X['x15'].unique())

CLF1 = LinearDiscriminantAnalysis()
model = CLF1.fit(X_train, y_train)
LDA_score = model.score(X_test, y_test)
print(LDA_score)
probs = model.predict_proba(X_test)
probs = probs[:, 1]
# print(probs)
auc = roc_auc_score(y_test, probs)
print(auc)

fpr, tpr, thresholds = roc_curve(y_test, probs)

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

plot_roc_curve(fpr,tpr)

#
# # KNN
#
# CLF3 = KNeighborsClassifier(n_neighbors=3)
# CLF3.fit(X_train, y_train)
# KNN_score3 = CLF3.score(X_test, y_test)
# print(KNN_score3)
#
# CLF4 = KNeighborsClassifier(n_neighbors=4)
# CLF4.fit(X_train, y_train)
# KNN_score4 = CLF4.score(X_test, y_test)
# print(KNN_score4)
#
# CLF5 = KNeighborsClassifier(n_neighbors=5)
# CLF5.fit(X_train, y_train)
# KNN_score5 = CLF5.score(X_test, y_test)
# print(KNN_score5)
#
# CLF6 = KNeighborsClassifier(n_neighbors=6)
# CLF6.fit(X_train, y_train)
# KNN_score6 = CLF6.score(X_test, y_test)
# print(KNN_score6)
#
# CLF7 = KNeighborsClassifier(n_neighbors=7)
# CLF7.fit(X_train, y_train)
# KNN_score7 = CLF7.score(X_test, y_test)
# print(KNN_score7)
#
# # highest
# CLF8 = KNeighborsClassifier(n_neighbors=8)
# CLF8.fit(X_train, y_train)
# KNN_score8 = CLF8.score(X_test, y_test)
# print(KNN_score8)
#
# CLF9 = KNeighborsClassifier(n_neighbors=9)
# CLF9.fit(X_train, y_train)
# KNN_score9 = CLF9.score(X_test, y_test)
# print(KNN_score9)
#
# highest
CLF10 = KNeighborsClassifier(n_neighbors=10)
CLF10.fit(X_train, y_train)
KNN_score10 = CLF10.score(X_test, y_test)
print(KNN_score10)
#
# ls = []
#
# for i in range(len(probs)):
#     if probs[i] > 0.8:
#         ls.append(1)
#     else:
#         ls.append(0)
#
#
#
# ls = pd.DataFrame(ls)
#
#
# cm = confusion_matrix(ls,y_test)
# print(cm)
