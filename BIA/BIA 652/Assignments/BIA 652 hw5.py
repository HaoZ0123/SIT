import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from scipy import cluster
from sklearn.cluster import KMeans

# Q1
print('Following is Q1:')
data = pd.read_excel('/Users/haodong/Desktop/Lung Function.xls',header=0)
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
y = np.ravel(y)
# print(y)

linked = linkage(X, 'ward')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)


plt.show()


# linked1 = linkage(X, 'complete')
# z = dendrogram(linked1, orientation='top', distance_sort='descending', show_leaf_counts=True)
# plt.title('Dendrogram')
# plt.show()


cutree = cluster.hierarchy.cut_tree(linked1,3)
l = np.ravel(cutree)

dat = {'Area':y, 'Cluster':l}
df = pd.DataFrame(dat)

t = pd.crosstab(df.Area, df.Cluster, margins=True)
print('\n', t,'\n')

# Q2
print('Following is Q2:')
kmeans = KMeans(n_clusters=4)
model = kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

dat1 = pd.DataFrame({'Area':y, 'Cluster':labels})

t2 = pd.crosstab(dat1.Area, dat1.Cluster, margins=True)
print('\n', t2, '\n')
