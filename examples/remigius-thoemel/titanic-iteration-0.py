# *********************************
# Initial Baseline
# *********************************


import numpy as np
from sklearn import preprocessing, neighbors
from sklearn import tree
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import sys

train_data = pd.read_csv('/vagrant/examples/data/titanic-train.csv')
solution_data = pd.read_csv('/vagrant/examples/data/titanic-test.csv')
min_max_scaler = preprocessing.MinMaxScaler()

def cat_to_binary(col_name, data, calculated_feature_list):
    col_data = data[col_name]
    categories = np.unique(col_data)
    for cat in categories:
        binary = (col_data == cat)
        new_column_name = col_name + '-' + str(cat)
        data[new_column_name] = binary.astype("int")
        calculated_feature_list.append(new_column_name)
    return

def preprocess(data):
    calculated_feature_list = []

    # Process Embarked
    data['Embarked'] = data['Embarked'].fillna('S')
    cat_to_binary('Embarked', data, calculated_feature_list)
    data.drop(['Embarked'], 1, inplace=True)

    # Process Pclass
    cat_to_binary('Pclass', data, calculated_feature_list)
    data.drop(['Pclass'], 1, inplace=True)

    # Process SibSp
    cat_to_binary('SibSp', data, calculated_feature_list)
    data.drop(['SibSp'], 1, inplace=True)

    # Process Parch
    cat_to_binary('Parch', data, calculated_feature_list)
    data.drop(['Parch'], 1, inplace=True)

    # Process Sex
    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1
    calculated_feature_list.append('Sex')

    data["Age"].fillna(data["Age"].median(), inplace=True)
    data["Fare"].fillna(data["Fare"].median(), inplace=True)
    data[['Age', 'Fare']] = min_max_scaler.fit_transform(data[['Age', 'Fare']])
    calculated_feature_list.append('Age')
    calculated_feature_list.append('Fare')
    return calculated_feature_list, data


def add_missing_columns(feat_list_to_be_transformed, feat_list_target, data):
    for column in feat_list_target:
        if column not in feat_list_to_be_transformed:
            data[column] = 0


def create_consolidated_feature_list(feat_list_1, feat_list_2):
    consolidated_feature_list = feat_list_1;
    for column in feat_list_2:
        if column not in feat_list_1:
            consolidated_feature_list.append(column)
    return consolidated_feature_list

# *********************************
# Preprocess Data
# *********************************

print('Initial Training Data :\n', train_data[:10])
train_feature_list, train_features = preprocess(train_data)
solution_feature_list, solution_features = preprocess(solution_data)


# Make sure training and solution features have same columns
add_missing_columns(train_feature_list, solution_feature_list, train_features)
add_missing_columns(solution_feature_list, train_feature_list, train_features)
print('Features After Preprocessing :\n', train_features[:10])
feature_list = create_consolidated_feature_list(train_feature_list, solution_feature_list)
print('Consolidated Feature list: ', train_feature_list)


train_labels = pd.DataFrame(train_data['Survived'])
print('Features After Preprocessing :\n', train_features[feature_list][:10])
print('Labels After Preprocessing :\n', train_labels[:10])


# *********************************
# Choose the model
# *********************************

# Linear Classifier
#clf = SGDClassifier(loss="modified_huber")

# Random forest
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)


# *********************************
#  Evaluate Model
# *********************************

X_train, X_test, y_train, y_test = train_test_split(train_features[feature_list], train_labels, test_size=0.1, random_state=0)

clf.fit(X_train, np.ravel(y_train))

# predict with propabilities, run on test data
preds = clf.predict_proba(X_test)
accuracy = clf.score(X_test, y_test)

# calculate roc and auc
fpr, tpr, thr = roc_curve(y_test, preds[:,1])
auc = roc_auc_score(y_test, preds[:,1])
print('AUC for linear classifier: ', auc)
print('Accuracy: ', accuracy)

# and plot it
plt.plot(fpr,tpr)
plt.plot(fpr,fpr)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
# plt.show()

fi = list(zip(feature_list, clf.feature_importances_))
fi.sort(key=lambda x: -x[1])
print('Feature Importance :', pd.DataFrame(fi, columns=["Feature","Importance"]))

# *********************************
# Calculate the solution
# *********************************
clf.fit(train_features[feature_list], np.ravel(train_labels))
print('Features After Preprocessing :\n', solution_features[feature_list][:10])

prediction = clf.predict(solution_features[feature_list])
print(prediction[:10])


solution = solution_data['PassengerId']
raw_data = {'PassengerId': solution.values, 'Survived': prediction}
df = pd.DataFrame(raw_data, columns = ['PassengerId', 'Survived'])
df.to_csv('/vagrant/examples/remigius-thoemel/titanic-solution-iteration-0.csv',index=False)
