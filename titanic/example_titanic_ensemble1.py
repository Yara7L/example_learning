import pandas as pd 
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.cross_validation import KFold
import numpy as np 
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier

titanic=pd.read_csv("E:/dataset/Titanic/train.csv")
titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())

# print(titanic["Sex"].unique())
titanic.loc[titanic["Sex"]=="male","Sex"]=1
titanic.loc[titanic["Sex"]=="female","Sex"]=0

# print(titanic["Embarked"].unique())
titanic["Embarked"]=titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"]=="S","Embarked"]=0
titanic.loc[titanic["Embarked"]=="C","Embarked"]=1
titanic.loc[titanic["Embarked"]=="Q","Embarked"]=2

titanic_test=pd.read_csv("E:/dataset/Titanic/test.csv")
titanic_test['Age'] = titanic_test['Age'].fillna(titanic['Age'].median())
titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median())
titanic_test.loc[titanic_test['Sex'] == 'male', 'Sex'] = 0
titanic_test.loc[titanic_test['Sex'] == 'female', 'Sex'] = 1
titanic_test['Embarked'] = titanic_test['Embarked'].fillna('S')
titanic_test.loc[titanic_test['Embarked'] == 'S', 'Embarked'] = 0
titanic_test.loc[titanic_test['Embarked'] == 'C', 'Embarked'] = 1
titanic_test.loc[titanic_test['Embarked'] == 'Q', 'Embarked'] = 2

'''
the new feature--FamilySize, NameLength
'''
titanic["FamilySize"]=titanic["SibSp"]+titanic["Parch"]
titanic["NameLength"]=titanic["Name"].apply(lambda x:len(x))


import re
def get_title(name):
    '''
    the new feature--Title, Master., Mr., Mrs.
    '''
    title_search=re.search('([A-Za-z]+)\.',name)
    if title_search:
        return title_search.group(1)
    return ""

titles=titanic["Name"].apply(get_title)
# print(pd.value_counts(titles))

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v
# print(pd.value_counts(titles))
titanic["Title"]=titles


import operator
family_id_mapping={}
def get_family_id(row):
    '''
    the new feature--FamilySize
    '''
    last_name=row["Name"].split(",")[0]
    family_id="{0}{1}".format(last_name,row["FamilySize"])

    if family_id not in family_id_mapping:
        if len(family_id_mapping)==0:
            current_id=1
        else:
            current_id=(max(family_id_mapping.items(),key=operator.itemgetter(1))[1]+1)
        family_id_mapping[family_id]=current_id
    return family_id_mapping[family_id]

family_ids=titanic.apply(get_family_id,axis=1)
family_ids[titanic["FamilySize"]<3]=-1

# print(pd.value_counts(family_ids))
titanic["FamilyId"]=family_ids

print("=============")
# print(np.isnan(titanic.any()))
# print(np.isnan(titanic_test.any()))
print(pd.isnull(titanic).any())
print(pd.isnull(titanic_test).any())

titles = titanic['Name'].apply(get_title)
print(pd.value_counts(titles))          # 输出看看, 相同数量的,设置相同映射

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, 
                 "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2 }
for k, v in title_mapping.items():
    titles[titles == k] = v

titanic_test['Title'] = titles
print(pd.value_counts(titanic_test['Title']))

titanic_test['FamilySize'] = titanic_test['SibSp'] + titanic_test['Parch']


predictors = ["Pclass","Sex","Age","Fare","Embarked","FamilySize","Title"]
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass","Sex","Age","Fare","Embarked","FamilySize","Title"]]
]

full_predictions = []
for alg, predictors in algorithms:
    alg.fit(titanic[predictors], titanic['Survived'])
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:, 1]
    full_predictions.append(predictions)

predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4

predictions[predictions<=.5]=0
predictions[predictions>.5]=1
predictions=predictions.astype(int)

submission=pd.DataFrame({
    "PassengerId":titanic_test["PassengerId"],
    "Survived":predictions
})

print(submission)