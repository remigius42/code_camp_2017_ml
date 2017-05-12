import numpy as np
from sklearn import preprocessing, neighbors
from sklearn import tree
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import KFold

# *********************************
# Use cross validation
# *********************************

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

    data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    cat_to_binary('Title', data, calculated_feature_list)

    data['Cabin_Prefix'] = data['Cabin'].str[0].fillna('Z')
    cat_to_binary('Cabin_Prefix', data, calculated_feature_list)

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

print('Initial Training Data :\n', train_data[:5])
train_feature_list, train_features = preprocess(train_data)
solution_feature_list, solution_features = preprocess(solution_data)


# Make sure training and solution features have same columns
add_missing_columns(train_feature_list, solution_feature_list, train_features)
add_missing_columns(solution_feature_list, train_feature_list, solution_features)
#print('Features After Preprocessing :\n', train_features[:10])
feature_list = create_consolidated_feature_list(train_feature_list, solution_feature_list)
print('Consolidated Feature list: ', train_feature_list)


train_labels = pd.DataFrame(train_data['Survived'])
#print('Features After Preprocessing :\n', train_features[feature_list][:10])
#print('Labels After Preprocessing :\n', train_labels[:10])


# *********************************
#  Evaluate Model
# *********************************

# Do cross validation and calculate mean for auc and accuracy
# https://stats.stackexchange.com/questions/186337/average-roc-for-repeated-10-fold-cross-validation-with-probability-estimates

tprs = []
aucs = []
accuracies = []
base_fpr = np.linspace(0, 1, 101)
plt.figure(figsize=(5, 5))

kf = KFold(n_splits=10)
i =0

for train, test in kf.split(train_features):
    print('KFold Iteration: ', i)
    clf = BaggingClassifier(RandomForestClassifier(n_estimators=100, n_jobs=-1),max_samples=0.5, max_features=0.5, random_state=0)
    clf.fit(train_features[feature_list].loc[train], np.array(train_labels.loc[train]).ravel())
    preds = clf.predict_proba(train_features[feature_list].loc[test])
    accuracy = clf.score(train_features[feature_list].loc[test], train_labels.loc[test])

    auc = roc_auc_score(train_labels.loc[test], preds[:, 1])
    fpr, tpr, _ = roc_curve(train_labels.loc[test], preds[:, 1])

    plt.plot(fpr, tpr, 'b', alpha=0.05)
    tpr = np.interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)
    aucs.append(auc)
    accuracies.append(accuracy)
    i += 1

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)

tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std

aucs = np.array(aucs)
print('Mean AUC: ', aucs.mean(axis=0), ', std dev: ', aucs.std(axis=0))

accuracies = np.array(accuracies)
print('Mean Accuracy: ', accuracies.mean(axis=0), ', std dev: ', accuracies.std(axis=0))


plt.plot(base_fpr, mean_tprs, 'b')
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.axes().set_aspect('equal', 'datalim')
plt.show()




# *********************************
# Calculate the submission
# *********************************
clf = BaggingClassifier(RandomForestClassifier(n_estimators=100, n_jobs=-1),max_samples=0.5, max_features=0.5, random_state=0)
clf.fit(train_features[feature_list], np.ravel(train_labels))
prediction = clf.predict(solution_features[feature_list])
#print(prediction[:10])


solution = solution_data['PassengerId']
raw_data = {'PassengerId': solution.values, 'Survived': prediction}
df = pd.DataFrame(raw_data, columns = ['PassengerId', 'Survived'])
df.to_csv('/vagrant/examples/remigius-thoemel/titanic-solution-iteration-5.csv',index=False)
