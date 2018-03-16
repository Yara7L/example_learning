#数据加载
import numpy as numpy
import urllib.request
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
raw_data = urllib.request.urlopen(url)
dataset = numpy.loadtxt(raw_data, delimiter=",")
print(dataset)
X = dataset[:,0:7]
y = dataset[:,8]

#数据标准化
from sklearn import preprocessing
normalized_x=preprocessing.normalize(X)
standardized_x=preprocessing.scale(X)
 
#特征的选取
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(X,y)
print('feature===================',model.feature_importances_)

#递归特征消除算法RFE:recursive feature elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
rfe=RFE(model,3)
rfe=rfe.fit(X,y)
print('rfe==========================',rfe.support_)
print('rfe rank  ',rfe.ranking_)

#逻辑回归
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X,y)
print(model)
expected=y
predicted=model.predict(X)
print("LogisticRegression:================")
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))

#朴素贝叶斯
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X,y)
print(model)
predicted=model.predict(X)
print("GaussionNaiveBayes:===================")
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))

#k-最近邻（KNN）
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
print("K-NeighborsN:===========================")
model.fit(X,y)
print(model)
expected=y
predicted=model.predict(X)
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))

#决策树
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X,y)
print("DecisionTree:==========================")
print(model)
expected=y
predicted=model.predict(X)
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))

#支持向量机SVN
from sklearn import metrics
from sklearn.svm import SVC
model=SVC()
model.fit(X,y)
print("SurportVectMachine:=====================")
print(model)
expected=y
predicted=model.predict(X)
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))

#规则化参数的选择
import numpy as numpy
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
alphas=numpy.array([1,0.1,0.01,0.001,0.0001,0])
model=Ridge()
grid=GridSearchCV(estimator=model,param_grid=dict(alpha=alphas))
grid.fit(X,y)
print("Ridge GridSearch:====================")
print(grid)
print(grid.best_score_)
print(grid.best_estimator_.alpha)

#随机从既定范围内选取参数更为高效
import numpy as numpy
from scipy.stats import uniform as sp_rand
from sklearn.linear_model import Ridge
from sklearn.grid_search import RandomizedSearchCV
param_grid={'alpha':sp_rand()}
model=Ridge()
rsearch=RandomizedSearchCV(estimator=model,param_distributions=param_grid,n_iter=100)
rsearch.fit(X,y)
print("Ridge RandomizedSearch:================")
print(rsearch)
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)