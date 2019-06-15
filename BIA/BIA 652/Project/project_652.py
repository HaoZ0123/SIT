### BIA 652 final project ###
### Group 10 ###

import pandas as pd
import numpy as np
import copy
import time
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing


##########  Data clean ##########

# import data
data = pd.read_csv('/Users/haodong/Desktop/652 project/dataset.csv')

# drop useless column
dat = data.iloc[:, 1:]

########## currency conversion ##########
# find unique currency
currency = dat.iloc[:, 4]
c_kind = pd.unique(currency)

# function for currency conversion
def currencyConversion(n, country):
    result = int(n)
    if country == 'USD':
        result = n * 1
    elif country == 'AUD':
        result = n * 0.71
    elif country == 'GBP':
        result = n * 1.3
    elif country == 'HKD':
        result = n * 0.13
    elif country == 'EUR':
        result = n * 1.21
    elif country == 'SEK':
        result = n * 0.11
    elif country == 'NOK':
        result = n * 0.12
    elif country == 'CAD':
        result = n * 0.75
    elif country == 'JPY':
        result = n * 0.01
    elif country == 'MXN':
        result = n * 0.05
    elif country == 'NZD':
        result = n * 0.68
    elif country == 'CHF':
        result = n * 1
    elif country == 'DKK':
        result = n * 0.15
    elif country == 'SGD':
        result = n * 0.74
    return result

dat1 = pd.DataFrame.copy(dat, deep=True)
goal = pd.DataFrame.copy(dat1.loc[:, 'goal'], deep=True)
pledged = pd.DataFrame.copy(dat1.loc[:, 'pledged'], deep=True)

for i in range(0, len(dat1)):
    goal[i] = round(currencyConversion(dat1.iloc[i]['goal'], dat1.iloc[i]['currency']))
    pledged[i] = round(currencyConversion(dat1.iloc[i]['pledged'], dat1.iloc[i]['currency']))
    dat1['goal'][i] = goal[i]
    dat1['pledged'][i] = pledged[i]
    dat1['currency'][i] = 'USD'

dat1 = pd.DataFrame.dropna(dat1, axis=0, how='any')
pd.DataFrame(dat1).to_excel('/Users/haodong/Desktop/652 project/BIA652 Data.xlsx')


########## Re-format time ##########
dat1 = pd.read_csv('/Users/haodong/Desktop/652 project/BIA652 Data.csv')
dat1 = dat1.iloc[:, 1:]

for i in range(0, len(dat1)):
    dat1['deadline'][i] = time.strftime("%Y-%m-%d", time.localtime(dat1.iloc[i]['deadline']))
    dat1['created_at'][i] = time.strftime("%Y-%m-%d", time.localtime(dat1.iloc[i]['created_at']))
    dat1['launched_at'][i] = time.strftime("%Y-%m-%d", time.localtime(dat1.iloc[i]['launched_at']))

# output new dataset with uniform currency and time format
pd.DataFrame(dat1).to_excel('/Users/haodong/Desktop/652 project/BIA652 Dataset.xlsx')

########## Re category ##########
data = pd.read_excel("/Users/haodong/Desktop/652 project/BIA652 Dataset.xlsx", header=0)

lo = data['loc_country'].unique()
cat = data['cate_name'].unique()
c = pd.DataFrame(cat)

# re category Countries
continent = []
for i in range(0, len(data.loc_country)):
    if data.loc_country[i] in ['HK', 'JP', 'YE', 'CN', 'TH', 'VN', 'TR']:
        continent.append('Asia')
    elif data.loc_country[i] in ['IT', 'SE', 'NO', 'ES', 'DK', 'GR', 'UA', 'RE', 'AT', 'AQ', 'RU', 'SI', 'EE', 'PL', 'PT', 'HR', 'SK', 'BY', 'LT', 'IS']:
        continent.append('South & Central Europe')
    elif data.loc_country[i] in ['AT', 'BE', 'CZ', 'DE', 'FR', 'IE', 'NL', 'CH', 'GB', 'US', 'CA']:
        continent.append('West Europe & North America')
    elif data.loc_country[i] in ['CR', 'MX', 'AR', 'BR', 'EC']:
        continent.append('South & Central America')
    elif data.loc_country[i] in ['AU', 'NZ', 'SG', 'ID', 'GU', 'TO']:
        continent.append('Oceania')
    else:
        continent.append('Africa')
data['continent'] = continent

# re category categories
category = []
for j in range(0, len(data.cate_name)):
    if data.cate_name[j] in ['Animation', 'Architecture', 'Art Books', 'Conceptual Art', 'Couture', 'Crochet', 'Dance', 'Digital Art', 'Documentary',\
        'Embroidery', 'Fashion', 'Fiction', 'Fine Art', 'Latin', 'Nonfiction', 'Painting', 'Performance Art', 'Performances', 'Photography', 'Pottery',\
        'Public Art', 'Science Fiction', 'Sculpture', 'Typography', 'Video Art']:
        category.append('Art')
    elif data.cate_name[j] in ['Audio', 'Mixed Media', 'Radio & Podcasts', 'Sound', 'Web', 'Webseries']:
        category.append('Broadcast')
    elif data.cate_name[j] in ['Accessories', 'Apparel', 'Bacon', 'Calendars', 'Candles', "Children's Books", 'Childrenswear', 'Drinks', 'Food',\
        'Footwear', 'Glass', 'Jewelry', 'Kids', 'Ready-to-wear', 'Restaurants', 'Shorts', 'Textiles', 'Wearables']:
        category.append('Commodity')
    elif data.cate_name[j] in ['3D Printing', 'Camera Equipment', 'DIY Electronics', 'Gadgets', 'Gaming Hardware', 'Hardware', 'Robots', 'Television']:
        category.append('Hardware')
    elif data.cate_name[j] in ['Civic Design', 'Design', 'DIY', 'Graphic Design', 'Illustration', 'Installations', 'Interactive Design',\
        'Playing Cards', 'Plays', 'Product Design', 'Translations', 'Woodworking']:
        category.append('Human Action')
    elif data.cate_name[j] in ['Action', 'Comedy', 'Comics', 'Drama', 'Film & Video', 'Horror', 'Movie Theaters', 'Narrative Film', 'Romance',\
        'Theater', 'Thrillers']:
        category.append('Movie')
    elif data.cate_name[j] in ['Blues', 'Classical Music', 'Country & Folk', 'Electronic Music', 'Hip-Hop', 'Indie Rock', 'Jazz', 'Metal',\
        'Music', 'Music Videos', 'Musical', 'Pop', 'Punk', 'R&B', 'Rock', 'World Music']:
        category.append('Music')
    elif data.cate_name[j] in ['Anthologies', 'Comic Books', 'Cookbooks', 'Graphic Novels', 'Journalism', 'Literary Journals', 'Nature', 'Periodicals', \
        'Photobooks', 'Poetry', 'Publishing', 'Stationery', 'Webcomics', 'Zines']:
        category.append('Publish')
    elif data.cate_name[j] in ['Apps', 'Games', 'Live Games', 'Makerspaces', 'Mobile Games', 'Software', 'Tabletop Games', 'Video Games']:
        category.append('Software')
    else:
        category.append('Others')

data['category'] = category

# Set dummy values
classMap_loc = {'West Europe & North America':0, 'Oceania':1, 'Asia':2, 'South & Central Europe':3, 'South & Central America':4, 'Africa':5}
data['continent'] = data['continent'].map(classMap_loc)

classMap_cat = {'Broadcast':0, 'Software':1, 'Art':2, 'Human Action':3, 'Hardware':4, 'Music':5, 'Publish':6, 'Movie':7, 'Commodity':8, 'Others':9}
data['category'] = data['category'].map(classMap_cat)

# output data for classification
pd.DataFrame(data).to_excel('/Users/haodong/Desktop/652 project/data_update_dummy.xlsx')

data = pd.read_excel('/Users/haodong/Desktop/652 project/data_update_dummy.xlsx', header=0)

y = data.iloc[:, -5]
X = data.iloc[:, [3, 9, 14, 15, 16, 17]]

# Data standardization
X_scale = preprocessing.StandardScaler().fit_transform(X)

# PCA dimension reduction test
pca = PCA(n_components=6)
pca.fit(X_scale)
var = pca.explained_variance_ratio_
var1 = np.cumsum(np.round(var, decimals=4)*100)

plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.plot(var1)

A = np.asmatrix(X_scale.T) * np.asmatrix(X_scale)
U, S, V = np.linalg.svd(A)
eigvals = S ** 2 / np.sum(S ** 2)

fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(6) +1
plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)

plt.title('Eigenvalues for Depression Data')
plt.xlabel('Principal Component Number')
plt.ylabel('Eigenvalue')
plt.show()

pca = PCA(n_components=5)
pca.fit(X_scale)
dat = pca.transform(X_scale)
dat = pd.DataFrame(dat)
dat['status'] = list(y)
print(dat)

pd.DataFrame(dat).to_csv('/Users/haodong/Desktop/652 project/652dat.csv', index=False)

########## Modeling ##########
# Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

LDA = LinearDiscriminantAnalysis()
LR = LogisticRegression()
KNN = KNeighborsClassifier()
NB = GaussianNB()

# LDA
print('\nLDA Classifier')
LDA = LinearDiscriminantAnalysis()
LDA = LDA.fit(X_train, y_train)
LDA_score = LDA.score(X_test, y_test)
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
LR_probs = LR.predict_proba(X_test)
LR_probs = LR_probs[:, 1]
LR_auc = roc_auc_score(y_test, LR_probs)
print(LR_auc)
LR_auc = round(LR_auc, 4)
print(LR_auc)
LR_fpr, LR_tpr, thresholds = roc_curve(y_test, LR_probs, pos_label=1)

# KNN
print('\nKNeighborsClassifier')
# for i in range(3, 11):
#     KNN = 'model_KNN_{}'.format(i)
#     KNN = KNeighborsClassifier(n_neighbors = i)
#     KNN.fit(X_train, np.ravel(y_train))
#     KNN_probs = KNN.predict(X_test)
#     # print(KNN_probs)
#     # KNN_probs = KNN_probs[:, 1]
#     KNN_auc = roc_auc_score(y_test, KNN_probs)
#     print('k = {}'.format(i), KNN_auc)

# When k = 5, knn has best performance

KNN = KNeighborsClassifier(n_neighbors = 5)
KNN = KNN.fit(X_train, np.ravel(y_train))
KNN_probs = KNN.predict_proba(X_test)
KNN_probs = KNN_probs[:, 1]
KNN_auc = roc_auc_score(y_test, KNN_probs)
print(KNN_auc)
KNN_fpr, KNN_tpr, thresholds = roc_curve(y_test, KNN_probs, pos_label=1)

# NB
print('\nNB Classifier')
NB = GaussianNB()
NB = NB.fit(X_train, y_train)
NB_score = NB.score(X_test, y_test)
NB_probs = NB.predict_proba(X_test)
NB_probs = NB_probs[:, 1]
NB_auc = roc_auc_score(y_test, NB_probs)
NB_auc = round(NB_auc, 4)
print(NB_auc)
NB_fpr, NB_tpr, thresholds = roc_curve(y_test, NB_probs, pos_label=1)

# models plot
plt.plot(LDA_fpr, LDA_tpr, color='orange', label='LDA_AUC = {}'.format(LDA_auc))
plt.plot(LR_fpr, LR_tpr, color='red', label='LR_AUC = {}'.format(LR_auc))
plt.plot(KNN_fpr, KNN_tpr, color='green', label='KNN_AUC = {}'.format(KNN_auc))
plt.plot(NB_fpr, NB_tpr, color='blue', label='NB_AUC = {}'.format(NB_auc))
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Ensemble
print('\n\n')
estimator = [('NB', NB), ('LR', LR), ('LDA', LDA), ('KNN', KNN)]
ensemble = VotingClassifier(estimator, voting='soft')
ensemble = ensemble.fit(X_train, y_train)
ensemble_score = ensemble.score(X_test, y_test)
print(ensemble_score)
ensemble_probs = ensemble.predict_proba(X_test)
ensemble_probs = ensemble_probs[:, 1]
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


# Classification by using PCA data
dat = pd.read_csv('/Users/haodong/Desktop/652 project/652dat.csv')
X = dat.iloc[:, :-1]
y = dat.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# LDA
print('\nLDA Classifier')
LDA = LinearDiscriminantAnalysis()
LDA = LDA.fit(X_train, y_train)
LDA_score = LDA.score(X_test, y_test)
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
LR_probs = LR.predict_proba(X_test)
LR_probs = LR_probs[:, 1]
LR_auc = roc_auc_score(y_test, LR_probs)
print(LR_auc)
LR_auc = round(LR_auc, 4)
print(LR_auc)
LR_fpr, LR_tpr, thresholds = roc_curve(y_test, LR_probs, pos_label=1)

# KNN
print('\nKNeighborsClassifier')
# for i in range(3, 11):
#     KNN = 'model_KNN_{}'.format(i)
#     KNN = KNeighborsClassifier(n_neighbors = i)
#     KNN.fit(X_train, np.ravel(y_train))
#     KNN_probs = KNN.predict(X_test)
#     # print(KNN_probs)
#     # KNN_probs = KNN_probs[:, 1]
#     KNN_auc = roc_auc_score(y_test, KNN_probs)
#     print('k = {}'.format(i), KNN_auc)

# When k = 3, knn has best performance

KNN = KNeighborsClassifier(n_neighbors = 3)
KNN = KNN.fit(X_train, np.ravel(y_train))
KNN_probs = KNN.predict_proba(X_test)
KNN_probs = KNN_probs[:, 1]
KNN_auc = roc_auc_score(y_test, KNN_probs)
print(KNN_auc)
KNN_fpr, KNN_tpr, thresholds = roc_curve(y_test, KNN_probs, pos_label=1)

# NB
print('\nNB Classifier')
NB = GaussianNB()
NB = NB.fit(X_train, y_train)
NB_score = NB.score(X_test, y_test)
NB_probs = NB.predict_proba(X_test)
NB_probs = NB_probs[:, 1]
NB_auc = roc_auc_score(y_test, NB_probs)
NB_auc = round(NB_auc, 4)
print(NB_auc)
NB_fpr, NB_tpr, thresholds = roc_curve(y_test, NB_probs, pos_label=1)

# models plot
plt.plot(LDA_fpr, LDA_tpr, color='orange', label='LDA_AUC = {}'.format(LDA_auc))
plt.plot(LR_fpr, LR_tpr, color='red', label='LR_AUC = {}'.format(LR_auc))
plt.plot(KNN_fpr, KNN_tpr, color='green', label='KNN_AUC = {}'.format(KNN_auc))
plt.plot(NB_fpr, NB_tpr, color='blue', label='NB_AUC = {}'.format(NB_auc))
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Ensemble
print('\n\n')
estimator = [('NB', NB), ('LR', LR), ('LDA', LDA), ('KNN', KNN)]
ensemble = VotingClassifier(estimator, voting='soft')
ensemble = ensemble.fit(X_train, y_train)
ensemble_score = ensemble.score(X_test, y_test)
print(ensemble_score)
ensemble_probs = ensemble.predict_proba(X_test)
ensemble_probs = ensemble_probs[:, 1]
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



