import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[143]:

# 导入数据
titanic = pd.read_csv('E:/dataset/Titanic/train.csv')
titanic.head(5)
# print(titanic.describe())


# In[144]:

titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
print(titanic.describe())


# In[145]:

print(titanic['Sex'].unique())

# Replace all the occurences of male with the number 0.
# 将字符值转换成 数值
# 进行一个属性值转换
titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1


# In[146]:

# 登船地址
print(titanic['Embarked'].unique())
titanic['Embarked'] = titanic['Embarked'].fillna('S')
titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2


# In[147]:

# Import the linear regression class (线性回归)
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation(交叉验证)
from sklearn.cross_validation import KFold

# The Columns we'll use to predict the target
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Initialize our algorithm class
alg = LinearRegression()
# Generate(生成) cross validation folds(交叉验证) for the titanic dataset.
# We set random_state to ensure we get the same splits(相同的分割) every time we run this.
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

# 预测结果
predictions = []
# 训练集, 测试集, 交叉验证
for train, test in kf:
    # The predictors we're using the train the algorithm. 
    # Note how we only take the rows in the train folds (只在训练集中进行)
    train_predictors = (titanic[predictors].iloc[train, :])
    # The target we're using to train the algorithm
    train_target = titanic['Survived'].iloc[train]
    # Training the algorithm using the prodictors and target
    # 训练数据的 X, Y ==> 让他能进行判断的操作
    alg.fit(train_predictors, train_target)
    # we can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test, :])
    predictions.append(test_predictions)


# In[148]:

import numpy as np

# The Predictions are in three separate numpy arrays. Concatenate them into one.
# We concatenate them on axis 0, as they only have one axis.我们将它们连接在轴0上，因为它们只有一个轴
predictions = np.concatenate(predictions, axis = 0)

# Map predictions to outcomes (only possible outcome are 1 and 0)
predictions[predictions > 0.5] = 1
predictions[predictions <= .5] = 0

# 进行评估模型
accuracy = sum(predictions[predictions == titanic['Survived']]) / len(predictions)
print(accuracy)


# In[149]:

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

# Initialize our algorithm
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds. (计算所有交叉验证折叠的精度分数。)
# (much simpler than what we did before !)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())


# ### 随机森林

# In[150]:

titanic_test = pd.read_csv('E:/dataset/Titanic/train.csv')
titanic_test['Age'] = titanic_test['Age'].fillna(titanic['Age'].median())
titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median())
titanic_test.loc[titanic_test['Sex'] == 'male', 'Sex'] = 0
titanic_test.loc[titanic_test['Sex'] == 'female', 'Sex'] = 1
titanic_test['Embarked'] = titanic_test['Embarked'].fillna('S')

titanic_test.loc[titanic_test['Embarked'] == 'S', 'Embarked'] = 0
titanic_test.loc[titanic_test['Embarked'] == 'C', 'Embarked'] = 1
titanic_test.loc[titanic_test['Embarked'] == 'Q', 'Embarked'] = 2


# In[151]:

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

#选中一些特征 
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Initialize our algorithm with the default paramters
# random_state = 1 表示此处代码多运行几次得到的随机值都是一样的，如果不设置，两次执行的随机值是不一样的
# n_estimators  指定有多少颗决策树，树的分裂的条件是:
# min_samples_split 代表样本不停的分裂，某一个节点上的样本如果只有2个了 ，就不再继续分裂了
# min_samples_leaf 是控制叶子节点的最小个数
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

# Compute the accuracy score for all the cross validation folds (nuch simpler than what we did before)
# 进行交叉验证 
kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=kf)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())


# In[152]:

# 建立100多个决策树
alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
# Compute the accuracy score
kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=kf)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())


# ## 关于特征提取问题 (非常关键)
# - 尽可能多的提取特征
# - 看不同特征的效果
# - 特征提取是数据挖掘里很- 要的一部分
# - 以上使用的特征都是数据里已经有的了，在真实的数据挖掘里我们常常没有合适的特征，需要我们自己取提取
# 

# In[153]:

# Generating a familysize column
# 合并数据 ：自己生成一个特征，家庭成员的大小:兄弟姐妹+老人孩子
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch']

# The .apply method generates a new series 名字的长度(据说国外的富裕的家庭都喜欢取很长的名字)
titanic['NameLength'] = titanic['Name'].apply(lambda x: len(x))


# In[154]:

import re

# A function to get the title from a name
def get_title(name):
    # Use a regular expression to search for a title.
    # Titles always consist of capital and lowercase letters.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Get all the titles and print how often each one occurs.
titles = titanic['Name'].apply(get_title)
print(pd.value_counts(titles))          # 输出看看, 相同数量的,设置相同映射

# 国外不同阶层的人都有不同的称呼
# Map each title to an integer. Some titles are very rare. and are compressed into the same codes as other
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, 
                 "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2 }
for k, v in title_mapping.items():
     #将不同的称呼替换成机器可以计算的数字
    titles[titles == k] = v
    
# Verify that we converted  everything
print(pd.value_counts(titles))

# Add in the title column
titanic['Title'] = titles


# In[155]:

# 进行特征选择
# 特征重要性分析
# 分析 不同特征对 最终结果的影响
# 例如 衡量age列的重要程度时，什么也不干，得到一个错误率error1，
# 加入一些噪音数据，替换原来的值(注意，此时其他列的数据不变)，又得到一个一个错误率error2
# 两个错误率的差值 可以体现这一个特征的重要性
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pylab as plt

# 选中一些特征
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', "Embarked",
             'FamilySize', 'Title', 'NameLength']

# Perform feature selection 选择特性
selector = SelectKBest(f_classif, k = 5)
selector.fit(titanic[predictors], titanic['Survived'])

# Get the raw p-values(P 值) for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores. See how "Plcass", "Sex", "Title", and "Fare" are the best ?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

# 通过以上的特征重要性分析, 选择出4个最重要的特性，重新进行随机森林的算法
# Pick only the four best features.
predictors = ['Pclass', 'Sex', 'Fare', 'Title']

alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)

# 进行交叉验证
kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"],cv=kf)
#目前的结果是没有得到提高，本处的目的是为了练习在随机森林中的特征选择，它对于实际的数据挖掘具有重要意义
print (scores.mean())


# ### 集成多种算法(减少过拟合)

# In[156]:

# 在竞赛中常用的耍赖的办法:集成多种算法，取最后每种算法的平均值，来减少过拟合
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# GradientBoostingClassifier也是一种随机森林的算法，可以集成多个弱分类器，然后变成强分类器
# The algorithm we want to ensemble
# We're using the more linear predictors for the logistic regression
# and everything with the gradient boosting
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),["Pclass","Sex","Age","Fare","Embarked","FamilySize","Title"]],
    [LogisticRegression(random_state=1), ["Pclass","Sex","Age","Fare","Embarked","FamilySize","Title"]]
]

# Initialize the cross validation folds
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = titanic['Survived'].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each folds
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(titanic[predictors].iloc[train, :], train_target)
        # Select and predict on the test fold.
        # The astype(float) is necessary to convert the dataframe
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test, :].astype(float))[:, 1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme - just average the predictions to get the final classification
    # 两个算法, 分别算出来的 预测值, 取平均
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over 5 is assumed to be a 1 prediction, and below 5 is a 0 prediction
    test_predictions[test_predictions <= 0.5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)
    
# Put all the predictions together into one array
predictions = np.concatenate(predictions, axis=0)

accuracy = sum(predictions[predictions == titanic['Survived']]) / len(predictions) 
print(accuracy)


# In[157]:

titles = titanic['Name'].apply(get_title)
print(pd.value_counts(titles))          # 输出看看, 相同数量的,设置相同映射

# 国外不同阶层的人都有不同的称呼
# Map each title to an integer. Some titles are very rare. and are compressed into the same codes as other
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, 
                 "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2 }
for k, v in title_mapping.items():
     #将不同的称呼替换成机器可以计算的数字
    titles[titles == k] = v
# Add in the title column
titanic_test['Title'] = titles
print(pd.value_counts(titanic_test['Title']))

# Now, we add the family size column.
titanic_test['FamilySize'] = titanic_test['SibSp'] + titanic_test['Parch']


# In[158]:

predictors = ["Pclass","Sex","Age","Fare","Embarked","FamilySize","Title"]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass","Sex","Age","Fare","Embarked","FamilySize","Title"]]
]

full_predictions = []
for alg, predictors in algorithms:
    # Fit the Algorithm using the full training data
    alg.fit(titanic[predictors], titanic['Survived'])
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:, 1]
    full_predictions.append(predictions)

# 梯度提升分类器产生更好的预测
# The gradient boosting classifier generates better predictions, so we weight it high
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
