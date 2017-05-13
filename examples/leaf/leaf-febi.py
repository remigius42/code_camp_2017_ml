# archive.ics.uci.edu/ml/datasets.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.metrics import accuracy_score, log_loss
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
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

def preprocess(data):
    # TODO
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

def encode(train, test):
    le = LabelEncoder().fit(train.species)
    labels = le.transform(train.species)           # encode species strings
    classes = list(le.classes_)                    # save column names for submission
    test_ids = test.id                             # save test ids for submission

    train = train.drop(['species', 'id'], axis=1)
    test = test.drop(['id'], axis=1)

    return train, labels, test, test_ids, classes


np.random.seed(42)

df_raw = pd.read_csv('/vagrant/examples/leaf/train.csv')
test_raw = pd.read_csv('/vagrant/examples/leaf/test.csv')

train, labels, test, test_ids, classes = encode(df_raw, test_raw)
#print(len(labels))


#printdata(df_raw, "BEFORE")
#df = preprocess(df_raw)
#test = preprocess(test_raw)
#printdata(df, "AFTER")


# divide data into train and test set
#X = np.array(df.drop(['species'],1))
#y = np.array(df['species'])
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, labels, test_size=0.2)

sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)
for train_index, test_index in sss:
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]


# Random Forest
rf1 = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf1.fit(X_train, y_train)
#print('Random Forest classifier: ', rf1)

print('**** Results ****')
train_predictions = rf1.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print("Accuracy: {:.4%}".format(acc))
train_proba_predictions = rf1.predict_proba(X_test)
ll = log_loss(y_test, train_proba_predictions)
print("Log Loss: {}".format(ll))


# Multi-Layer Perceptron
mlpc = MLPClassifier(hidden_layer_sizes=(164,), solver='lbfgs', max_iter=500, momentum=0.2)
mlpc.fit(X_train, y_train)

print('**** Results ****')
train_predictions = mlpc.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print("Accuracy: {:.4%}".format(acc))
train_proba_predictions = mlpc.predict_proba(X_test)
ll = log_loss(y_test, train_proba_predictions)
print("Log Loss: {}".format(ll))


# XXX temp finish here
#exit(0)


# compose solution to challenge
# prediction = svc.predict(test)
test_predictions = mlpc.predict_proba(test)
print("Test Predictions: size = ", len(test_predictions))
print(test_predictions)
#
# Format DataFrame
submission = pd.DataFrame(test_predictions, columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()

# Export Submission
submission.to_csv('submission-febi.csv', index = False)
#print(submission.tail())
