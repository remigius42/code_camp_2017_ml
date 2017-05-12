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
# print(train.head(3))
# print(test.head(3))
# print(test_ids)
# print(classes)
print(len(labels))


#printdata(df_raw, "BEFORE")
#df = preprocess(df_raw)
#test = preprocess(test_raw)
#printdata(df, "AFTER")


# XXX temp finish here
#exit(0)


# divide data into train and test set
#X = np.array(df.drop(['species'],1))
#y = np.array(df['species'])
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, labels, test_size=0.2)

sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)
for train_index, test_index in sss:
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

# KNeighbors algorithm
# clf = neighbors.KNeighborsClassifier()
# clf.fit(X_train, y_train)
# accuracy = clf.score(X_test, y_test)
# print("KNeighbors - accuracy: ", accuracy)

# calculate roc and auc
# prediction_p = clf.predict_proba(X_test)
# print("Prediction Probabilities")
# print(prediction_p[:,1])
# plot_data(y_test, prediction_p[:,1], "KNeighbors")


# Linear classifier
# sgd = SGDClassifier(loss="modified_huber")
# sgd.fit(X_train, y_train)
# print('Linear classifier: ', sgd)

# calculate roc and auc
# prediction_p = sgd.predict_proba(X_test)
# print("Prediction Probabilities")
# print(prediction_p[:,1])
# plot_data(y_test, prediction_p[:,1], "Linear Classifier")


# Random Forest
rf1 = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf1.fit(X_train, y_train)
#print('Random Forest classifier: ', rf1)

# calculate predictions with probabilities
#prediction_p = rf1.predict_proba(X_test)
#plot_data(y_test, prediction_p[:,1], "Random Forest")

print('**** Results ****')
train_predictions = rf1.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print("Accuracy: {:.4%}".format(acc))
# print('predictions', len(train_predictions), train_predictions)
# print('y-test', len(y_test), y_test)

train_proba_predictions = rf1.predict_proba(X_test)
# print('proba_predictions', len(train_proba_predictions), train_proba_predictions)
print('y-test', len(y_test), y_test)
ll = log_loss(y_test, train_proba_predictions)
print("Log Loss: {}".format(ll))


# SVC
# svc = SVC(probability = True)
# svc.fit(X_train, y_train)
# #Y_pred = svc.predict(X_test)
# acc_svc = round(svc.score(X_train, y_train) * 100, 2)
# print('SVC- accuracy: ', acc_svc)

# calculate predictions with probabilities
# prediction_p = svc.predict_proba(X_test)
# plot_data(y_test, prediction_p[:,1], "SVC")


# compose solution to challenge
# prediction = svc.predict(test)
test_predictions = rf1.predict_proba(test)
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

# solution = test_original['PassengerId']
# raw_data = {'PassengerId': solution.values, 'Survived': prediction}
#
# df = pd.DataFrame(raw_data, columns = ['PassengerId', 'Survived'])
# df.to_csv('/vagrant/examples/data/titanic-febi-solution.csv',index=False)
