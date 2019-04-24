import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import datasets
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
##########
df = pd.read_csv("bank.csv")
df.head()
df.isnull().sum()
stat = df.describe()
g = sns.boxplot(x = df["age"])
sns.distplot(df.age, bins=100)
g = sns.boxplot(df["duration"])
sns.distplot(df.duration, bins=100)
##############
data = df.copy()
jobs = ['management','blue-collar','technician','admin.','services','retired','self-employed','student','unemployed','entrepreneur','housemaid','unknown']
for j in jobs:
    print("{:15} : {:5}". format(j, len(data[(data.deposit == "yes") & (data.job ==j)])))
data.job.value_counts()
################
data['job'] = data['job'].replace(['management', 'admin.'], 'white-collar')
data['job'] = data['job'].replace(['services','housemaid'], 'pink-collar')
data['job'] = data['job'].replace(['retired', 'student', 'unemployed', 'unknown'], 'other')
data.job.value_counts()
data.poutcome.value_counts()
###############
data['poutcome'] = data['poutcome'].replace(['other'] , 'unknown')
data.poutcome.value_counts()
##############
data.drop('contact', axis=1, inplace=True)
#############
data['default']
data['default_cat'] = data['default'].map({'yes':1, 'no':0})
data.drop("default", axis=1, inplace=True)
#################
data["housing_cat"]=data['housing'].map({'yes':1, 'no':0})
data.drop('housing', axis=1,inplace = True)
###############
data["loan_cat"] = data['loan'].map({'yes':1, 'no':0})
data.drop('loan', axis=1, inplace=True)
##############
data.drop('month', axis=1, inplace=True)
data.drop('day', axis=1, inplace=True)
################
data["deposit_cat"] = data['deposit'].map({'yes':1, 'no':0})
data.drop('deposit', axis=1, inplace=True)
#############
len(data[data.pdays == -1])
data['pdays'].max()
data.loc[data['pdays'] == -1, 'pdays'] = 10000
###############
data['recent_pdays'] = np.where(data['pdays'], 1/data.pdays, 1/data.pdays)
data.drop('pdays', axis=1, inplace=True)
data.tail()
################
dumy = pd.get_dummies(data= data, columns=['job', 'marital', 'education', 'poutcome'], prefix=['job', 'marital', 'education', 'poutcome'])
dumy.head()
dumy.shape
dumy.describe()
dumy.plot(kind = "scatter", x ='age', y = 'balance')
stat_1 = data[data.deposit_cat == 1].describe()
##############
len(dumy[(dumy.deposit_cat == 1) & (dumy.default_cat == 1)])
plt.figure(figsize=(10,6))
sns.barplot(x = 'job', y = 'deposit_cat', data = data)
plt.figure(figsize=(10,6))
sns.barplot(x = 'poutcome', y = 'duration', data = data)
############
df2 = dumy
corr = df2.corr()
plt.figure(figsize = (10,10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .82})
plt.title('Heatmap of Correlation Matrix')
cor_dep = pd.DataFrame(corr['deposit_cat'].drop('deposit_cat'))
cor_dep.sort_values(by = 'deposit_cat', ascending=True)
#############
X = df2.drop('deposit_cat', 1)
y = df2.deposit_cat
data_train, data_test, label_train, label_test = train_test_split(X, y, test_size = 0.2, random_state = 50)
##############
dt2 = tree.DecisionTreeClassifier(max_depth=2)
dt2.fit(data_train, label_train)
dt2.score(data_train, label_train)
dt2.score(data_test, label_test)
###############
dt3 = tree.DecisionTreeClassifier(max_depth=3)
dt3.fit(data_train, label_train)
dt3.score(data_train, label_train)
dt3.score(data_test, label_test)
###############
dt4 = tree.DecisionTreeClassifier(max_depth=4)
dt4.fit(data_train, label_train)
dt4.score(data_train, label_train)
dt4.score(data_test, label_test)
############
dt6 = tree.DecisionTreeClassifier(max_depth=6)
dt6.fit(data_train, label_train)
dt6.score(data_train, label_train)
dt6.score(data_test, label_test)
###########
dt1 = tree.DecisionTreeClassifier()
dt1.fit(data_train, label_train)
dt1.score(data_train, label_train)
dt1.score(data_test, label_test)
###########
feat = df2.columns.tolist()
feat
###########
dt2 = tree.DecisionTreeClassifier(max_depth=2)
dt2.fit(data_train, label_train)
fi = dt2.feature_importances_
l = len(feat)
for i in range(0, len(feat)):
    print('{:.<20} {:3}'.format(feat[i],fi[i]))
############
dumy.duration.mean()
dumy.duration.max()
dumy.duration.min()
##############
print(dt2.predict_proba(np.array([46,3354,522,1,1,0,1,0,0.005747,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0]).reshape(1, -1)))
##########
preds = dt2.predict(data_test)
metrics.accuracy_score(label_test,preds)
metrics.roc_auc_score(label_test, preds)
################################################
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["job_code"] = le.fit_transform(data['job'])
data["mat_code"] = le.fit_transform(data['marital'])
data["edu_code"] = le.fit_transform(data['education'])
data["poc_code"] = le.fit_transform(data['poutcome'])
df3 = data.drop(['job', 'marital', 'education','poutcome'], axis=1)
X = df3.drop('deposit_cat', 1)
y = df3.deposit_cat
data_train, data_test, label_train, label_test = train_test_split(X, y, test_size = 0.2, random_state = 50)
##############
dt2 = tree.DecisionTreeClassifier(max_depth=2)
dt2.fit(data_train, label_train)
dt2.score(data_train, label_train)
dt2.score(data_test, label_test)
###############
dt3 = tree.DecisionTreeClassifier(max_depth=3)
dt3.fit(data_train, label_train)
dt3.score(data_train, label_train)
dt3.score(data_test, label_test)
###############
dt4 = tree.DecisionTreeClassifier(max_depth=4)
dt4.fit(data_train, label_train)
dt4.score(data_train, label_train)
dt4.score(data_test, label_test)
############
dt6 = tree.DecisionTreeClassifier(max_depth=6)
dt6.fit(data_train, label_train)
dt6.score(data_train, label_train)
dt6.score(data_test, label_test)
###########
dt1 = tree.DecisionTreeClassifier()
dt1.fit(data_train, label_train)
dt1.score(data_train, label_train)
dt1.score(data_test, label_test)