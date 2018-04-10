# -*- coding:utf-8 -*-
import numpy as np 
import pandas as pd 
import xgboost as xgb 
import math 
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.preprocessing import Imputer,LabelEncoder
from sklearn import cross_validation,metrics
import matplotlib.pyplot as plt

def data_preprocessing():
    '''
    preprocess the data
    '''
    train=pd.read_csv("E:/dataset/Titanic/train.csv")
    test=pd.read_csv("E:/dataset/Titanic/test.csv")
    submission=pd.read_csv("E:/dataset/Titanic/gender_submission.csv")
    print(train.shape,test.shape)
    count=submission.loc[submission["Survived"]==1].count()   
    print(count)

    sex_count=train.groupby(['Sex','Survived'])['Survived'].count()
    print(sex_count)

    train['Sex']=train['Sex'].map({'female':0,'male':1}).astype(int)
    test['Sex']=test['Sex'].map({'female':0,'male':1}).astype(int)

    embark_dummies=pd.get_dummies(train['Embarked'])
    train=train.join(embark_dummies)
    train.drop(['Embarked','PassengerId'],axis=1,inplace=True)
    train['Fare_Category']=train['Fare'].map(fare_category)
    columns=train.columns

    embark_dummies_test=pd.get_dummies(test['Embarked'])
    test=test.join(embark_dummies_test)
    test.drop(['Embarked','PassengerId'],axis=1,inplace=True)
    test['Fare_Category']=test['Fare'].map(fare_category)
    columns_test=test.columns

    for f in train.columns:
        if train[f].dtype=='object':
            label=LabelEncoder()
            label.fit(list(train[f].values))
            train[f]=label.transform(list(train[f].values))

    for f in test.columns:
        if test[f].dtype=='object':
            label=LabelEncoder()
            label.fit(list(test[f].values))
            test[f]=label.transform(list(test[f].values))

    print("the columns which is nan")
    na_train=train.isnull().sum().sort_values(ascending=False)

    na_test=test.isnull().sum().sort_values(ascending=False)
    print(na_train)
    print(na_test)

    train_data=train.values
    imput=Imputer(missing_values='NaN',strategy="mean",axis=0)
    imput=imput.fit(train_data)
    train_data=imput.fit_transform(train_data)

    test_data=test.values
    imput_test=Imputer(missing_values='NaN',strategy="mean",axis=0)
    imput_test=imput_test.fit(test_data)
    test_data=imput_test.fit_transform(test_data)

    train=pd.DataFrame(train_data,index=None,columns=columns)
    na_train=train.isnull().sum().sort_values(ascending=False)

    test=pd.DataFrame(test_data,index=None,columns=columns_test)
    na_test=test.isnull().sum().sort_values(ascending=False)
    print("after:")
    print(na_train)
    print(na_test)
    print(train.head(1))
    print(test.head(1))

    train.to_csv("E:/dataset/Titanic/train_data.csv")
    test.to_csv("E:/dataset/Titanic/test_data.csv")

def fare_category(fare):
    if fare <= 4:
        return 0
    elif fare <= 10:
        return 1
    elif fare <=30:
        return 2
    elif fare <= 45:
        return 3
    else:
        return 4


def load_data():
    train_data=pd.read_csv("E:/dataset/Titanic/train_data.csv")
    test_data=pd.read_csv("E:/dataset/Titanic/test_data.csv")

    X=train_data.drop(['Survived'],1)
    y=train_data['Survived']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=7)

    return X_train,X_test,y_train,y_test,test_data

def train_logisticR():
    X_train,X_test,y_train,y_test,test_data=load_data()

    model=LogisticRegression(penalty='l2')
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    rfc_rate,rmse=calc_accuracy(y_pred,y_test)
    total=total_survived(y_pred)

    # predictions=model.predict(test_data)
    # pre_total=total_survived(predictions)
    # print(pre_total)


    return rfc_rate,rmse,total

def calc_accuracy(y_pred,y_true):
    accuracy=metrics.accuracy_score(y_true,y_pred)
    rmse=np.sqrt(np.mean((y_pred-y_true)**2))
    return accuracy,rmse

def total_survived(y_pred):
    total=0
    for value in y_pred:
        if value==1:
            total+=1
    return total

def train_randomforest():
    X_train,X_test,y_train,y_test=load_data()
    model=RandomForestClassifier(n_estimators=500,max_depth=6,random_state=7)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    rfc_rate,rmse=calc_accuracy(y_pred,y_test)
    total=total_survived(y_pred)
    return rfc_rate,rmse,total

def train_XGBoost():
    X_train,X_test,y_train,y_test=load_data()
    model=xgb.XGBClassifier(max_depth=8,learning_rate=0.06,n_estimators=100,
                            objective="binary:logistic",silent=True,min_child_weight=6)
    eval_data=[(X_test,y_test)]
    model.fit(X_train,y_train,eval_set=eval_data,early_stopping_rounds=30)

    y_pred=model.predict(X_test)
    rfc_rate,rmse=calc_accuracy(y_pred,y_test)
    total=total_survived(y_pred)

    return rfc_rate,rmse,total

def train():
    lg_rate,lg_rmse,lg_total=train_logisticR()
    rf_rate,rf_rmse,rf_total=train_randomforest()
    xg_rate,xg_rmse,xg_total=train_XGBoost()

    print("LogisticRegression acc_rate:{0:.4f},RMS:{1:.4f},存活：{2}".format(lg_rate,lg_rmse,lg_total))
    print("RandomForest acc_rate:{0:.4f},RMS:{1:.4f},存活：{2}".format(rf_rate,rf_rmse,rf_total))
    print("XGB acc_rate:{0:.4f},RMS{1:.4f},存活：{2}".format(xg_rate,xg_rmse,xg_total))

def optimized_XGBoost():
    X_train,X_test,y_train,y_test=load_data()
    model=xgb.XGBClassifier(max_depth=7,n_estimators=300,min_child_weight=5,
                            colsample_bytree=0.6,subsample=0.9,reg_alpha=0.005)
    eval_data=[(X_test,y_test)]
    model.fit(X_train,y_train,eval_set=eval_data,early_stopping_rounds=30)
    y_pred=model.predict(X_test)
    acc,rmse=calc_accuracy(y_pred,y_test)

    print("accuracy:{0:.2f}%".format(100*acc))

def cv_XGBoost():
    X_train,X_test,y_train,y_test,test_data=load_data()
    model=xgb.XGBClassifier(max_depth=7,n_estimators=300,min_child_weight=5,
                            colsample_bytree=0.6,subsample=0.9,reg_alpha=0.005)
    grid_params={
        'reg_alpha':[0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6],
        'max_depth': range(2, 7, 1)
    }
    gcv=GridSearchCV(estimator=model,param_grid=grid_params)
    gcv.fit(X_train,y_train)
    # print('----',gcv.grid_scores_)
    print('====',gcv.best_params_)
    print("Accuracy:{0:.4f}%".format(100*gcv.best_score_))
    predicts=gcv.predict(test_data)

# def predit():
#     load_data()


if __name__=='__main__':
    # data_preprocessing()
    # # load_data()
    # train()
    # # optimized_XGBoost()
    cv_XGBoost()

