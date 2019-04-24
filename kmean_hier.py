import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import make_blobs

df = make_blobs(n_samples=200, n_features=2, centers=4,cluster_std=1.6, random_state=50)

points = df[0]
points

plt.scatter(df[0][:,0],df[0][:,1], c = df[1], cmap='viridis')

from sklearn.cluster import KMeans
means = KMeans(n_clusters=4)
means.fit(points)
print(means.cluster_centers_)
pred = means.fit_predict(points)


plt.scatter(points[pred ==0,0], points[pred == 0,1], s=100, c='red')
plt.scatter(points[pred ==1,0], points[pred== 1,1], s=100, c='black')
plt.scatter(points[pred ==2,0], points[pred== 2,1], s=100, c='blue')
plt.scatter(points[pred ==3,0], points[pred == 3,1], s=100, c='cyan')

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

den = sch.dendrogram(sch.linkage(points, method='ward'))
hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(points)

plt.scatter(points[y_hc ==0,0], points[y_hc == 0,1], s=100, c='red')
plt.scatter(points[y_hc==1,0], points[y_hc == 1,1], s=100, c='black')
plt.scatter(points[y_hc ==2,0], points[y_hc == 2,1], s=100, c='blue')
plt.scatter(points[y_hc ==3,0], points[y_hc == 3,1], s=100, c='cyan')
###################################
df = pd.read_csv("Iris.csv")
df
X = df.drop(["Id", "Species"], axis=1)
y = df[["Species"]]

X = X.as_matrix()

den = sch.dendrogram(sch.linkage(X, method='ward'))
hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)
y_hc

plt.scatter(X[y_hc ==0,0], X[y_hc == 0,1], s=50, c='red')
plt.scatter(X[y_hc ==1,0], X[y_hc == 1,1], s=50, c='black')
plt.scatter(X[y_hc ==2,0], X[y_hc == 2,1], s=50, c='blue')
plt.scatter(X[y_hc ==3,0], X[y_hc == 3,1], s=50, c='cyan')

means = KMeans(n_clusters=3)
pred = means.fit_predict(X)

plt.scatter(X[pred ==0,0], X[pred == 0,1], s=50, c='red')
plt.scatter(X[pred ==1,0], X[pred == 1,1], s=50, c='black')
plt.scatter(X[pred ==2,0], X[pred == 2,1], s=50, c='blue')

sns.scatterplot(df.SepalLengthCm,df.SepalWidthCm, hue=df.Species)