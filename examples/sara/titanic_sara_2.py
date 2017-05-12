# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# machine learning
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import SVC, LinearSVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB

np.random.seed(0)

def analyse_data():
    print('1. Find out which columns are present')
    print(train_df.columns.values)
    # ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare' 'Cabin' 'Embarked']

    print('2. See an example of the values')
    train_df.head()
    train_df.tail()

    print('3. See infos about datatypes of train and test')
    train_df.info()
    print('_' * 40)
    test_df.info()

    print('4. See infos about value distribution')
    print(train_df.describe())
    print(train_df.describe(include=['O']))

    print('5. Analyzing Data Math')
    for feature in train_df.columns.values:
        if feature in ['Survived', 'Name', 'Cabin', 'Ticket', 'Cabin']:
            continue
        print('Feature', feature)
        info = train_df[[feature, 'Survived']].groupby([feature], as_index=False).mean().sort_values(by='Survived',
                                                                                                     ascending=False)
        print(info)

    print('6. Age analysis')
    plt.figure(1)
    plt.subplot(211)
    age_survivors = train_df.loc[train_df['Survived'] == 1, 'Age']
    plt.hist(age_survivors.dropna(), 10, normed=1, facecolor='green', alpha=0.75)
    plt.xlabel('Age')
    plt.ylabel('Survivors')
    plt.title('Age Analysis')
    plt.plot()

    plt.subplot(212)
    age_dead = train_df.loc[train_df['Survived'] == 0, 'Age']
    plt.hist(age_dead.dropna(), 10, normed=1, facecolor='green', alpha=0.75)
    plt.xlabel('Age')
    plt.ylabel('Deaths')
    plt.plot()

    print('7. Fare analysis')
    plt.figure(2)
    plt.subplot(211)
    fare_survivors = train_df.loc[train_df['Survived'] == 1, 'Fare']
    plt.hist(fare_survivors.dropna(), 10, normed=1, facecolor='green', alpha=0.75)
    plt.xlabel('Fare')
    plt.ylabel('Survivors')
    plt.title('Fare Analysis')
    plt.plot()

    plt.subplot(212)
    fare_dead = train_df.loc[train_df['Survived'] == 0, 'Fare']
    plt.hist(fare_dead.dropna(), 10, normed=1, facecolor='green', alpha=0.75)
    plt.xlabel('Age')
    plt.ylabel('Deaths')
    plt.plot()
    plt.show()


def drop_irrelevant_columns(train_df, test_df):
    print('1. Drop columns that have no correlation to Survivors')
    train_df = train_df.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    return train_df, test_df


def pre_title(train_df, test_df):
    print('2. Extract title feature')
    combine = [train_df, test_df]
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    title_analysis = pd.crosstab(train_df['Title'], train_df['Sex'])
    #print(title_analysis)

    # rare titles
    rare_titles_df = title_analysis.where(title_analysis['female'] + title_analysis['male'] < 10).dropna()
    rare_titles = list(rare_titles_df.index)

    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

        dataset['Title'] = dataset['Title'].replace(rare_titles, 'Rare')

    #print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
        dataset['Title'] = dataset['Title'].astype(int)

    train_df = train_df.drop(['Name'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)

    return train_df, test_df

def pre_sex(train, test):
    print('3. Converting sex feature')
    train['Sex'] = train_df['Sex'].map({'female': 1, 'male': 0}).astype(int)
    test['Sex'] = test_df['Sex'].map({'female': 1, 'male': 0}).astype(int)
    return train, test

def pre_age_fill_gaps(train, test):
    print('4. Age Fill Gaps')
    guess_ages = np.zeros((2, 3))

    combine = [train, test]
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) &
                                   (dataset['Pclass'] == j + 1)]['Age'].dropna()

                age_guess = guess_df.median()

                # Convert random age float to nearest .5 age
                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1),
                            'Age'] = guess_ages[i, j]

        dataset['Age'] = dataset['Age'].astype(int)
    return train, test


def pre_age(train, test):
    print('5. Age create age bands')
    train['AgeBand'] = pd.cut(train_df['Age'], 5)
    train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand',
                                                                                              ascending=True)
    combine = [train, test]
    for dataset in combine:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    train = train.drop(['AgeBand'], axis=1)
    return train, test


def pre_family_size(train, test):
    print('6. Add family size feature')
    combine = [train, test]
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    #print(train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

    train = train.drop(['SibSp', 'Parch'], axis=1)
    test = test.drop(['SibSp', 'Parch'], axis=1)

    return train, test


def pre_port(train, test):
    print('7. port of origin')
    most_frequent_port = train_df.Embarked.dropna().mode()[0]

    combine = [train, test]
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(most_frequent_port)
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    return train, test


def pre_fare(train, test):
    print('8. Fare')
    train['Fare'].fillna(train['Fare'].dropna().median(), inplace=True)
    test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)

    train['FareBand'] = pd.qcut(train['Fare'], 5)
    #print(train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

    combine = [train, test]
    for dataset in combine:
        dataset.loc[dataset['Fare'] <= 7.854, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare'] = 2
        dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare'] = 3
        dataset.loc[dataset['Fare'] > 39.688, 'Fare'] = 4
        dataset['Fare'] = dataset['Fare'].astype(int)

    train = train.drop(['FareBand'], axis=1)
    print(train.head())
    return train, test


def preprocess_data(train_df, test_df):
    print('PREPROCESSING DATA')

    train_df, test_df = drop_irrelevant_columns(train_df, test_df)
    train_df, test_df = pre_title(train_df, test_df)
    train_df, test_df = pre_sex(train_df, test_df)
    train_df, test_df = pre_age_fill_gaps(train_df, test_df)
    train_df, test_df = pre_age(train_df, test_df)
    train_df, test_df = pre_family_size(train_df, test_df)
    train_df, test_df = pre_port(train_df, test_df)
    train_df, test_df = pre_fare(train_df, test_df)

    return train_df, test_df


def plot_curve(fpr, tpr, title):
    plt.plot(fpr, tpr)
    plt.plot(fpr, fpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    plt.show()


def train_logistic_regression(x_train, y_train, x_cv, y_cv):
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    preds = clf.predict_proba(x_cv)
    fpr, tpr, thr = roc_curve(y_cv, preds[:, 1])
    auc = roc_auc_score(y_cv, preds[:, 1])
    print('AUC for logistic regression: ', auc)
    
    plot_curve(fpr, tpr, 'Logistic regression ' + str(auc))
    return clf


def train_svm(x_train, y_train, x_cv, y_cv):
    clf = SVC(probability=True)
    clf.fit(x_train, y_train)

    preds = clf.predict_proba(x_cv)
    fpr, tpr, thr = roc_curve(y_cv, preds[:, 1])
    auc = roc_auc_score(y_cv, preds[:, 1])
    print('AUC for support vector machine: ', auc)

    #plot_curve(fpr, tpr, 'SVM ' + str(auc))
    return clf


def analyze_correlation(clf, method):
    print('Analysis of correlation', method)
    coeff_df = pd.DataFrame(train_df.columns.delete(0))
    coeff_df.columns = ['Feature']
    coeff_df["Correlation"] = pd.Series(clf.coef_[0])

    print(coeff_df.sort_values(by='Correlation', ascending=False))


def predict_test(clf):
    X_test = test_df.drop('PassengerId', axis=1)
    prediction = clf.predict(X_test)

    solution = test_df['PassengerId']
    raw_data = {'PassengerId': solution.values, 'Survived': prediction}
    df = pd.DataFrame(raw_data, columns = ['PassengerId', 'Survived'])
    df.to_csv('/vagrant/examples/sara/titanic-solution-iteration-2.csv', index=False)


train_df = pd.read_csv('data/titanic-train.csv')
test_df = pd.read_csv('data/titanic-test.csv')

#analyse_data()
train_df, test_df = preprocess_data(train_df, test_df)

print('START TRAINING CLASSIFIERS')
train_train_df, train_cv_df = model_selection.train_test_split(train_df, test_size=0.2)

x_train = train_train_df.drop('Survived', axis=1)
y_train = train_train_df['Survived']
x_cv = train_cv_df.drop('Survived', axis=1)
y_cv = train_cv_df['Survived']

#clf = train_logistic_regression(x_train, y_train, x_cv, y_cv)
#analyze_correlation(clf, 'Logistic Regression')

clf = train_svm(x_train, y_train, x_cv, y_cv)

predict_test(clf)




