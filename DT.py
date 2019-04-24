import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from sklearn.model_selection import learning_curve, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
###############
from sklearn.datasets import load_breast_cancer
df = load_breast_cancer()
data = np.c_[df.data, df.target]
col = np.append(df.feature_names, ["target"])
df1 = pd.DataFrame(data , columns= col)
df1.head()
X = df1[df1.columns[:-1]]
y = df1.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
X.columns.values

g = sns.heatmap(X_train.corr(), cmap="BrBG", annot=False)


from sklearn.decomposition import PCA
X3 = X
y3 = y
var = 5
pca = PCA(n_components=5)
X_tran = pca.fit_transform(X3)
pca.explained_variance_
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)
X3pca = pd.DataFrame(X_tran, y3)
X3pca.head()


clf1 = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=12)
clf1.fit(X_train, y_train)
pred = clf1.predict(X_train)
accuracy_score(y_train, pred)

names = X.columns.values
def plot_decision_tree1(a,b):
    dot_data = tree.export_graphviz(a, out_file=None,
                             feature_names=b,
                             class_names=['Malignant','Benign'],
                             filled=False, rounded=True,
                             special_characters=False)
    graph = graphviz.Source(dot_data)
    return graph
plot_decision_tree1(clf1, names)

def plot_feature_importances(clf, feature_names):
    c_features = len(feature_names)
    plt.barh(range(c_features), clf.feature_importances_)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature name")
    plt.yticks(np.arange(c_features), feature_names)
plot_feature_importances(clf1, names)

pred2 = clf1.predict(X_test)
accuracy_score(y_test, pred2)