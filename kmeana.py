import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("cars.csv")
df.head()
X = df.iloc[:,:-1]
X = pd.DataFrame(X)
X = X.convert_objects(convert_numeric=True)
X.columns = ['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60', 'year']

for i in X.columns:
    X[i] = X[i].fillna(int(X[i].mean()))

X.isnull().sum()

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters=3)
y_means = kmeans.fit_predict(X)

print(kmeans.labels_)
print(kmeans.cluster_centers_)

plt.scatter(X[:,0], X[:,1], c = kmeans.labels_, cmap="rainbow")

plt.scatter(X[y_means == 0, 0], X[y_means == 0,1],s=100,c='red',label='US')
plt.scatter(X[y_means == 1, 0], X[y_means == 1,1],s=100,c='blue',label='Japan')
plt.scatter(X[y_means == 2, 0], X[y_means == 2,1],s=100,c='green',label='Europe')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of car brands')
plt.legend()
###############################################################################################
df = pd.read_csv("Iris.csv")
df.head()
X = df.iloc[:,[1,2,3,4]].values
df.iloc[:, 5]

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

Kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_means = kmeans.predict(X)

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
plt.legend()

y_means
