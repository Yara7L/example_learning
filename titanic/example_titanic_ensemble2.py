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
print(pd.value_counts(titles))

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v
print(pd.value_counts(titles))
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

print(pd.value_counts(family_ids))
titanic["FamilyId"]=family_ids

# from sklearn.ensemble import RandomForestClassifier 
# from sklearn.feature_selection import SelectKBest, f_classif
# import matplotlib.pyplot as plt

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

# # SelectKBest is to select features, don't konw the inner.
# selector = SelectKBest(f_classif, k=10)
# selector.fit(titanic[predictors], titanic["Survived"])

# scores = -np.log10(selector.pvalues_)

# plt.bar(range(len(predictors)), scores)
# plt.xticks(range(len(predictors)), predictors, rotation='vertical')
# plt.show()

# # select the 4 highest scores
# predictors = ["Pclass", "Sex", "Fare", "Title"]
# alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
# kf=cross_validation.KFold(titanic.shape[0],n_folds=3,random_state=1)
# scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"])
# print("randomforest is",scores.mean())



from sklearn.ensemble import GradientBoostingClassifier
# limit the deep and the numbers of the trees

# ensemble. if the model isn't terrible, the more the models differ, the better the last model is.
# (the different features bring the result)
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []

for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []

    for alg, predictors in algorithms:
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
   
    test_predictions = (full_test_predictions[0]*3 + full_test_predictions[1]) / 4
    
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print("GradientBoosting+LogisticRegression",accuracy)
