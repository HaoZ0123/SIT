import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

# import data here


data = pd.read_excel("/Users/haodong/Desktop/data_update_dummy.xlsx", header=0)



y = data.iloc[:, -3]
X = data.iloc[:, [2,8,13,14,20,21]]




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

LDA = LinearDiscriminantAnalysis()
LR = LogisticRegression()
KNN = KNeighborsClassifier()
NB = GaussianNB()
MNB = MultinomialNB()

# MNB
MNB = MultinomialNB()
MNB.fit(X_train, y_train)
MNB_score = MNB.score(X_test, y_test)
print(MNB_score)
MNB_probs = MNB.predict_proba(X_test)
MNB_probs = MNB_probs[:, 1]
print(MNB_probs)
MNB_auc = roc_auc_score(y_test, MNB_probs)
MNB_auc = round(MNB_auc, 4)
print(MNB_auc)

MNB_fpr, MNB_tpr, thresholds = roc_curve(y_test, MNB_probs, pos_label=1)


# LDA
print('\nLDA Classifier')
LDA = LinearDiscriminantAnalysis()
LDA = LDA.fit(X_train, y_train)
LDA_score = LDA.score(X_test, y_test)
# print(LDA_score)
LDA_probs = LDA.predict_proba(X_test)
LDA_probs = LDA_probs[:, 1]
LDA_auc = roc_auc_score(y_test, LDA_probs)
LDA_auc = round(LDA_auc, 4)
print(LDA_auc)

LDA_fpr, LDA_tpr, thresholds = roc_curve(y_test, LDA_probs, pos_label=1)

# LR
print('\nLR Classifier')
LR = LogisticRegression(solver='lbfgs', max_iter=5000)
LR = LR.fit(X_train, y_train)
LR_score = LR.score(X_test, y_test)
# print(LR_score)
LR_probs = LR.predict_proba(X_test)
LR_probs = LR_probs[:, 1]
# print(probs)
LR_auc = roc_auc_score(y_test, LR_probs)
print(LR_auc)
LR_auc = round(LR_auc, 4)
print(LR_auc)

LR_fpr, LR_tpr, thresholds = roc_curve(y_test, LR_probs, pos_label=1)


# SVM
print('\nSVM Classifier')
SVM = svm.SVC(kernel='rbf', probability=True)
SVM = SVM.fit(X_train, y_train)
SVM_score = SVM.score(X_test, y_test)
# print(SVM_score)
SVM_probs = SVM.predict_proba(X_test)
SVM_probs = SVM_probs[:, 1]
# print(SVM_probs)
SVM_auc = roc_auc_score(y_test, SVM_probs)
SVM_auc = round(SVM_auc, 4)
print(SVM_auc)

SVM_fpr, SVM_tpr, thresholds = roc_curve(y_test, SVM_probs, pos_label=1)


# NB
print('\nNB Classifier')
NB = GaussianNB()
NB = NB.fit(X_train, y_train)
NB_score = NB.score(X_test, y_test)
# print(NB_score)
NB_probs = NB.predict_proba(X_test)
NB_probs = NB_probs[:, 1]
# print(NB_probs)
NB_auc = roc_auc_score(y_test, NB_probs)
NB_auc = round(NB_auc, 4)
print(NB_auc)

NB_fpr, NB_tpr, thresholds = roc_curve(y_test, NB_probs, pos_label=1)

# models plot
plt.plot(LDA_fpr, LDA_tpr, color='orange', label='LDA_AUC = {}'.format(LDA_auc))
plt.plot(LR_fpr, LR_tpr, color='red', label='LR_AUC = {}'.format(LR_auc))
plt.plot(SVM_fpr, SVM_tpr, color='green', label='SVM_AUC = {}'.format(SVM_auc))
plt.plot(NB_fpr, NB_tpr, color='blue', label='NB_AUC = {}'.format(NB_auc))
plt.plot(MNB_fpr, MNB_tpr, color='black', label='MNB_AUC = {}'.format(MNB_auc))
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Ensemble
print('\n\n')
estimator = [('NB', NB), ('LR', LR), ('LDA', LDA), ('SVM', SVM), ('MNB', MNB)]
ensemble = VotingClassifier(estimator, voting='soft')
ensemble = ensemble.fit(X_train, y_train)
ensemble_score = ensemble.score(X_test, y_test)
print(ensemble_score)
ensemble_probs = ensemble.predict_proba(X_test)
ensemble_probs = ensemble_probs[:, 1]
# print(NB_probs)
ensemble_auc = roc_auc_score(y_test, ensemble_probs)
print(ensemble_auc)
ensemble_auc = round(ensemble_auc, 4)
print(ensemble_auc)

ensemble_fpr, ensemble_tpr, thresholds = roc_curve(y_test, ensemble_probs, pos_label=1)

# ensemble model plot
plt.plot(ensemble_fpr, ensemble_tpr, color='orange', label='ensemble_AUC = {}'.format(ensemble_auc))
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()




