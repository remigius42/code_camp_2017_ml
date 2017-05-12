# archive.ics.uci.edu/ml/datasets.html

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
import os

df_raw = pd.read_csv('/vagrant/examples/data/titanic-train.csv')
test_raw = pd.read_csv('/vagrant/examples/data/titanic-test.csv')

test_original = pd.DataFrame.copy(test_raw)

def preprocess(data):
    print("********************************")
    print("Before")
    print("********************************")
    print(data.head())
    print(data.describe())
    data.replace('?', -99999, inplace=True)
    data.drop(['PassengerId'], 1, inplace=True)
    data.drop(['Name'], 1, inplace=True)
    data.drop(['Ticket'], 1, inplace=True)
    data.drop(['Cabin'], 1, inplace=True)

    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1


    data["Embarked"] = data["Embarked"].fillna("S")
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2
    
    print("********************************")
    print("After")
    print("********************************")
    print(data.head())
    print(data.describe())
    
    return data


df = preprocess(df_raw)
test = preprocess(test_raw)


X = np.array(df.drop(['Survived'],1))
y = np.array(df['Survived'])


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

prediction = clf.predict(test)
print(len(prediction))


solution = test_original['PassengerId']
raw_data = {'PassengerId': solution.values, 'Survived': prediction}

df = pd.DataFrame(raw_data, columns = ['PassengerId', 'Survived'])

df.to_csv('/vagrant/examples/data/titanic-solution.csv',index=False)

#accuracy = clf.score(X_test, y_test)
#print(accuracy)
