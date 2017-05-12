# archive.ics.uci.edu/ml/datasets.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import doctest


def printdata(data, title):
    print()
    print("********************************")
    print(title)
    print("********************************")
    print(data.head())
    print(data.describe())

def extract_title(name):
    idx = name.find(',')
    name = name[(idx+1):]

    idx = name.find('.')
    name = name[:idx]

    return name.strip()

def map_title(title):
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    return title.map(title_mapping)

def hash_string(str):
    return abs(hash(str)) % (10 ** 4)

def preprocess(data):
    # why this?!
    data.replace('?', -99999, inplace=True)

    # drop columns we don't want (yet)
    data.drop(['PassengerId'], 1, inplace=True)
    #data.drop(['Name'], 1, inplace=True)
    data.drop(['Ticket'], 1, inplace=True)
    data.drop(['Cabin'], 1, inplace=True)

    # guess missing values, do we really want that?!
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    data["Embarked"] = data["Embarked"].fillna("S")

    # drop rows with missing values
    data = data.dropna()

    # convert float to int columns (is this a good idea?)
    data["Age"] = data["Age"].apply(lambda x : int(x))
    data["Fare"] = data["Fare"].apply(lambda x : int(x))

    # transform enum to int columns
    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1

    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2

    # extract titles
    data["Title"] = data["Name"].apply(lambda x : extract_title(x))
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data["Title"] = data["Title"].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)
    data["Title"] = data["Title"].apply(lambda x : int(x))
    #data["Name"] = data["Name"].apply(lambda x : hash_string(extract_title(x)))
    data.drop(['Name'], 1, inplace=True)

    return data

def plot_data(values, probabilities, algorithm):
    fpr, tpr, thr = roc_curve(values, probabilities)
    auc = roc_auc_score(values, probabilities)
    print(' AUC for ', algorithm, ': ', auc)

    # and plot it
    plt.plot(fpr, tpr)
    plt.plot(fpr, fpr)
    plt.title(algorithm)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.show()


np.random.seed(42)

df_raw = pd.read_csv('/vagrant/examples/data/titanic-train.csv')
test_raw = pd.read_csv('/vagrant/examples/data/titanic-test.csv')
test_original = pd.DataFrame.copy(test_raw)

printdata(df_raw, "BEFORE")

df = preprocess(df_raw)
test = preprocess(test_raw)

printdata(df, "AFTER")

# XXX temp finish here
#exit(0)


# divide data into train and test set
X = np.array(df.drop(['Survived'],1))
y = np.array(df['Survived'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


# KNeighbors algorithm
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("KNeighbors - accuracy: ", accuracy)

# calculate roc and auc
prediction_p = clf.predict_proba(X_test)
print("Prediction Probabilities")
print(prediction_p[:,1])

plot_data(y_test, prediction_p[:,1], "KNeighbors")


# Linear classifier
sgd = SGDClassifier(loss="modified_huber")
sgd.fit(X_train, y_train)
print('Linear classifier: ', sgd)

# calculate roc and auc
prediction_p = sgd.predict_proba(X_test)
print("Prediction Probabilities")
print(prediction_p[:,1])

plot_data(y_test, prediction_p[:,1], "Linear Classifier")


# Random Forest
rf1 = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf1.fit(X_train, y_train)
print('Random Forest classifier: ', rf1)

# calculate predictions with probabilities
prediction_p = rf1.predict_proba(X_test)
plot_data(y_test, prediction_p[:,1], "Random Forest")


# SVC
svc = SVC(probability = True)
svc.fit(X_train, y_train)
#Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 2)
print('SVC- accuracy: ', acc_svc)

# calculate predictions with probabilities
prediction_p = svc.predict_proba(X_test)
plot_data(y_test, prediction_p[:,1], "SVC")


# compose solution to challenge
prediction = svc.predict(test)
print("Prediction: size = ", len(prediction))
print(prediction)

solution = test_original['PassengerId']
raw_data = {'PassengerId': solution.values, 'Survived': prediction}

df = pd.DataFrame(raw_data, columns = ['PassengerId', 'Survived'])
df.to_csv('/vagrant/examples/data/titanic-febi-solution.csv',index=False)
