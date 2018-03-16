import pandas as pd 
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.cross_validation import KFold
import numpy as np 
from sklearn import cross_validation

titanic=pd.read_csv("E:/dataset/Titanic/train.csv")
titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())

print(titanic["Sex"].unique())
titanic.loc[titanic["Sex"]=="male","Sex"]=1
titanic.loc[titanic["Sex"]=="female","Sex"]=0

print(titanic["Embarked"].unique())
titanic["Embarked"]=titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"]=="S","Embarked"]=0
titanic.loc[titanic["Embarked"]=="C","Embarked"]=1
titanic.loc[titanic["Embarked"]=="Q","Embarked"]=2


from sklearn.ensemble import RandomForestClassifier

predictors=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# 提高随机森林预测精度，就是增加决策树的数量，调整min_sample_split和min_samples_leaf可以减少过拟合，
# 有助于让算法适应新数据，但是在训练集上的得分会降低。
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

kf=cross_validation.KFold(titanic.shape[0],n_folds=3,random_state=1)
scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=kf)
print(scores.mean())

# #params  fine-tuning
# alg2=RandomForestClassifier(random_state=1,n_estimators=50,min_samples_split=4,min_samples_leaf=2)
# kf=cross_validation.KFold(titanic.shape[0],n_folds=3,random_state=1)
# scores=cross_validation.cross_val_score(alg2,titanic[predictors],titanic["Survived"],cv=kf)
# print(scores.mean())