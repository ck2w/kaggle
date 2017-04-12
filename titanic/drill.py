__author__ = 'ken.chen'


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, tree, svm
import seaborn as sns
sns.set_style('whitegrid')

def data_clean(df):
    # df.set_index('PassengerId', inplace=True)

    df.ix[df['Sex'] == 'male', 'Sex'] = 1
    df.ix[df['Sex'] == 'female', 'Sex'] = 0

    df.ix[df['Embarked'] == 'S', 'Embarked'] = 0
    df.ix[df['Embarked'] == 'C', 'Embarked'] = 1
    df.ix[df['Embarked'] == 'Q', 'Embarked'] = 2

    df['Age'] = df['Age'].apply(lambda x: df['Age'].mean() if np.isnan(x) else x)

    df['Fare'] = df['Fare'].apply(lambda x: df['Fare'].mean() if np.isnan(x) else x)

    return df

df_train = pd.read_csv('train.csv')
df_train.set_index('PassengerId', inplace=True)
df_train = df_train[['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

df_train.ix[df_train['Sex'] == 'male', 'Sex'] = 1
df_train.ix[df_train['Sex'] == 'female', 'Sex'] = 0

df_train.ix[df_train['Embarked'] == 'S', 'Embarked'] = 0
df_train.ix[df_train['Embarked'] == 'C', 'Embarked'] = 1
df_train.ix[df_train['Embarked'] == 'Q', 'Embarked'] = 2

df_train['Age'] = df_train['Age'].apply(lambda x: df_train['Age'].mean() if np.isnan(x) else x)
df_train.dropna(axis=0, inplace=True)



reg = linear_model.LogisticRegression()
# clf = tree.DecisionTreeClassifier()
# clf = svm.SVC()

reg.fit(df_train[['Sex', 'Age', 'Fare', 'Embarked', 'Parch', 'SibSp', 'Pclass']], df_train['Survived'])
# clf.fit(df_train[['Sex', 'Age', 'Fare', 'Embarked', 'Parch', 'SibSp', 'Pclass']], df_train['Survived'])

# test data

df_test = pd.read_csv('test.csv')
df_test = data_clean(df_test)

df_test['Survived'] = reg.predict(df_test[['Sex', 'Age', 'Fare', 'Embarked', 'Parch', 'SibSp', 'Pclass']])
# df_test['Survived'] = clf.predict(df_test[['Sex', 'Age', 'Fare', 'Embarked', 'Parch', 'SibSp', 'Pclass']])

df_test = df_test[['PassengerId', 'Survived']]

df_test.to_csv('submission.csv', index=False)


print 'end'