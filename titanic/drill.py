__author__ = 'ken.chen'


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, tree, svm

def data_clean(df):
    # df.set_index('PassengerId', inplace=True)

    df['Sex'][df['Sex'] == 'male'] = 1
    df['Sex'][df['Sex'] == 'female'] = 0

    df['Embarked'][df['Embarked'] == 'S'] = 0
    df['Embarked'][df['Embarked'] == 'C'] = 1
    df['Embarked'][df['Embarked'] == 'Q'] = 2

    df['Age'] = df['Age'].apply(lambda x: 35.67 if np.isnan(x) else x)

    df['Fare'] = df['Fare'].apply(lambda x: df['Fare'].mean() if np.isnan(x) else x)

    return df

df_train = pd.read_csv('train.csv')

df_train.set_index('PassengerId', inplace=True)

df_train = df_train[['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

df_train['Sex'][df_train['Sex'] == 'male'] = 1
df_train['Sex'][df_train['Sex'] == 'female'] = 0

df_train['Embarked'][df_train['Embarked'] == 'S'] = 0
df_train['Embarked'][df_train['Embarked'] == 'C'] = 1
df_train['Embarked'][df_train['Embarked'] == 'Q'] = 2

df_train['Age'] = df_train['Age'].apply(lambda x: 35.67 if np.isnan(x) else x)

df_train.dropna(axis=0, inplace=True)


reg = linear_model.LogisticRegression()
# clf = tree.DecisionTreeClassifier()
clf = svm.SVC()

reg.fit(df_train[['Sex', 'Age', 'Fare', 'Embarked', 'Parch', 'SibSp', 'Pclass']], df_train['Survived'])
clf.fit(df_train[['Sex', 'Age', 'Fare', 'Embarked', 'Parch', 'SibSp', 'Pclass']], df_train['Survived'])

# test data

df_test = pd.read_csv('test.csv')
df_test = data_clean(df_test)

df_test['Survived'] = reg.predict(df_test[['Sex', 'Age', 'Fare', 'Embarked', 'Parch', 'SibSp', 'Pclass']])
# df_test['Survived'] = clf.predict(df_test[['Sex', 'Age', 'Fare', 'Embarked', 'Parch', 'SibSp', 'Pclass']])

df_test = df_test[['PassengerId', 'Survived']]

df_test.to_csv('submission.csv', index=False)


print 'end'