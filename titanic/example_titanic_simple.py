import pandas as pd 
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.cross_validation import KFold
import numpy as np 
from sklearn import cross_validation

titanic=pd.read_csv("E:/dataset/Titanic/train.csv")
print(titanic.head(5))

# mam is filled by the median
titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())

print(titanic["Sex"].unique())
titanic.loc[titanic["Sex"]=="male","Sex"]=1
titanic.loc[titanic["Sex"]=="female","Sex"]=0

print(titanic["Embarked"].unique())
titanic["Embarked"]=titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"]=="S","Embarked"]=0
titanic.loc[titanic["Embarked"]=="C","Embarked"]=1
titanic.loc[titanic["Embarked"]=="Q","Embarked"]=2

# the features to train
predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]

# LinearRegression
alg=LinearRegression()

# k-fold validation
kf=KFold(titanic.shape[0],n_folds=3,random_state=1)

predictions=[]
for train,test in kf:
    train_predictors=(titanic[predictors].iloc[train,:])
    train_target=titanic["Survived"].iloc[train]
    alg.fit(train_predictors,train_target)
    test_predictions=alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)


predictions=np.concatenate(predictions,axis=0)

predictions[predictions>.5]=1
predictions[predictions<=.5]=0
accuracy=len(predictions[predictions==titanic["Survived"]])/len(predictions)
print(accuracy)


# LogisticRegression
alg=LogisticRegression(random_state=1)

scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=3)
print(scores.mean())


titles = titanic_test["Name"].apply(get_title)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
for k,v in title_mapping.items():
    titles[titles == k] = v
titanic_test["Title"] = titles

print(pd.value_counts(titanic_test["Title"]))

titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]

print(family_id_mapping)

family_ids = titanic_test.apply(get_family_id, axis=1)
family_ids[titanic_test["FamilySize"] < 3] = -1
titanic_test["FamilyId"] = family_ids
titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))



predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

full_predictions = []
for alg, predictors in algorithms:
    alg.fit(titanic[predictors], titanic["Survived"])
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)
# predict_proba（）,得出得分，predict（），直接是分类结果

predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4

predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1

predictions = predictions.astype(int)

submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })