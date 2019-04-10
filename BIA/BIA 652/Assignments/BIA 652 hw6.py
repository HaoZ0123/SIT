import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt

data = pd.read_excel('/Users/haodong/Desktop/Classification Data2.xlsx', header=0)
# print(data)

X = data.iloc[:, :-1]
# print(X)

X_scale = preprocessing.StandardScaler().fit_transform(X)
# X_scale = pd.DataFrame(X_scale,columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12'])
# print(X_scale)

pca = PCA(n_components=12)
pca.fit(X_scale)
X_scale1 = pca.transform(X_scale)
var = pca.explained_variance_ratio_
var1 = np.cumsum(np.round(var * 100, decimals=4))

plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.plot(var1)

cov = np.cov(X_scale.T)
evs, evc = np.linalg.eig(cov)
print(evs)
print(evc[0:2])


fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(12) + 1
plt.plot(sing_vals, evs, 'ro-', linewidth=2)


plt.title('Eigenvalues for Depression Data')
plt.xlabel('Principal Component Number')
plt.ylabel('Eigenvalue')
# plt.show()

# pca1 = PCA(n_components=6)
# pca1.fit(X_scale)
# X_scale2 = pca1.transform(X_scale)
# cov1 = np.cov(X_scale2.T)
# evs1, evc1 = np.linalg.eig(cov1)
# print(evs1)
# print(evc1)
#


