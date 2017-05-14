#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
import sklearn
from sklearn.model_selection import cross_val_score, train_test_split

from collections import Counter
import pickle
import os

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold



np.random.seed(0)

#﻿export PYTHONIOENCODING=UTF-8

feature_columns = ['Summary', 'Issue Type', 'Priority', 'Reporter', 'Creator', 'Created', 'Affects Version/s', 'Images', 'Environment', 'Description', 'Resolution']
lower_bound_w_count = 5
upper_bound_w_count = 100
min_word_len = 4
summary_lex_file = '/vagrant/examples/data/summary_lex_iteration_1.pickle'
description_lex_file = '/vagrant/examples/data/description_lex_iteration_1.pickle'
preprocessed_file = '/vagrant/examples/data/preprocessed_features_iteration_1.pickle'



def cat_to_binary(col_name, data):
    col_data = data[col_name]
    categories = np.unique(col_data)
    for cat in categories:
        binary = (col_data == cat)
        new_column_name = col_name + '-' + str(cat)
        data[new_column_name] = binary.astype("int")
    return




def build_lexicon_for_column(column_name):
    lexicon = []
    word_tokens = []
    for index, row in alm_data.iterrows():
        # print(row[column_name])
        word_tokens = word_tokenize(str(row[column_name]))
        lexicon += list(word_tokens)

    print('number of words in column ', column_name, ': ', len(lexicon))

    lexicon = [stemmer.stem(str(w)) for w in lexicon]
    w_counts = Counter(lexicon)
    print('number of distinct words: ', len(w_counts))


    filtered_w_counts = [w for w in w_counts if not w in stop_words_de]
    print('number of filtered distinct words: ', len(filtered_w_counts))

    lexicon_with_relevant_words = []

    for word in filtered_w_counts:
        if upper_bound_w_count > w_counts[word] > lower_bound_w_count:
            if len(word) >= min_word_len:
                lexicon_with_relevant_words.append(word)

    return lexicon_with_relevant_words


def create_summary_column_name(word):
    return 'summary_' + str(word)


def create_description_column_name(word):
    return 'description_' + str(word)


if os.path.exists(preprocessed_file):
    f = open(preprocessed_file, 'rb')
    alm_data = pickle.load(f)
    f.close()
    print('Alm_data loaded from pickle', preprocessed_file)
else:

    print('Pickle for alm_data ', preprocessed_file, 'does not exist yet')

    raw_data = pd.read_excel('/vagrant/examples/data/ALMsINCs.xlsx')
    #alm_data[feature_columns].to_csv('/vagrant/examples/data/ALMforWeka.csv',index=False)
    alm_data = raw_data


    stop_words_de = set(stopwords.words('german'))
    stemmer = SnowballStemmer("german")
    # stemmer = PorterStemmer()

    print('Initial shape after loading: ', alm_data.shape)
    alm_data = alm_data[((alm_data['Resolution'] == 'Fixed') | (alm_data['Resolution'] == 'Won\'t Fix'))]
    print('Shape after filtering for Fixed/WontFix: ', alm_data.shape)
    alm_data = alm_data[feature_columns]
    print('Shape after reducing to columns known at submit time: ', alm_data.shape)

    alm_data.loc[alm_data["Issue Type"] == "ALM", "Issue Type"] = 0
    alm_data.loc[alm_data["Issue Type"] == "INC", "Issue Type"] = 1

    cat_to_binary('Priority', alm_data)
    alm_data.drop(['Priority'], 1, inplace=True)

    cat_to_binary('Reporter', alm_data)
    alm_data.drop(['Reporter'], 1, inplace=True)

    cat_to_binary('Creator', alm_data)
    alm_data.drop(['Creator'], 1, inplace=True)

    alm_data.drop(['Affects Version/s'], 1, inplace=True)

    alm_data.loc[alm_data["Images"].notnull(), "Images"] = 0
    alm_data.loc[alm_data["Images"].isnull(), "Images"] = 1

    alm_data.drop(['Environment'], 1, inplace=True)
    alm_data.drop(['Created'], 1, inplace=True)

    if os.path.exists(summary_lex_file):
        f = open(summary_lex_file, 'rb')
        summary_lex = pickle.load(f)
        f.close()
        print('Summary_lex_file loaded from pickle', summary_lex_file)
    else:
        summary_lex = build_lexicon_for_column('Summary')
        with open(summary_lex_file,'wb') as f:
            pickle.dump(summary_lex, f, pickle.HIGHEST_PROTOCOL)

    if os.path.exists(description_lex_file):
        f = open(description_lex_file, 'rb')
        description_lex = pickle.load(f)
        f.close()
        print('Description_lex_file loaded from pickle', description_lex_file)
    else:
        description_lex = build_lexicon_for_column('Description')
        with open(description_lex_file,'wb') as f:
            pickle.dump(description_lex, f, pickle.HIGHEST_PROTOCOL)


    print(summary_lex.head())

    print(description_lex.head())

    summary_columns = []
    description_columns = []


    def add_columns_for_word_dictionary(df, word_dict):
        columns = []
        for word in word_dict:
            df[create_description_column_name(word)] = 0
            columns.append(create_description_column_name(word))
        print("New shape:", alm_data.shape)
        return df, columns


    # Comment out in order to measure without these columns
    _, summary_columns = add_columns_for_word_dictionary(alm_data, summary_lex)
    _, description_columns = add_columns_for_word_dictionary(alm_data, description_lex)


    print(alm_data.shape)

    for index, row in alm_data.iterrows():
        word_tokens = word_tokenize(str(row['Summary']))
        stemmed_word_tokens = [stemmer.stem(str(w)) for w in word_tokens]
        for word in stemmed_word_tokens:
            if create_summary_column_name(word) in summary_columns:
                row[create_summary_column_name(word)] += 1
        word_tokens = word_tokenize(str(row['Description']))
        stemmed_word_tokens = [stemmer.stem(str(w)) for w in word_tokens]
        for word in stemmed_word_tokens:
            if create_description_column_name(word) in summary_columns:
                row[create_description_column_name(word)] += 1

    alm_data.drop(['Summary'], 1, inplace=True)
    alm_data.drop(['Description'], 1, inplace=True)

    with open(preprocessed_file,'wb') as f:
        pickle.dump(alm_data, f, pickle.HIGHEST_PROTOCOL)
        print('alm_data stored to pickle ', preprocessed_file)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()


# *********************************
#  Evaluate Model
# *********************************

alm_data['Resolution_int'] = 0
alm_data.loc[alm_data["Resolution"] == "Fixed", "Resolution_int"] = 1
alm_data.loc[alm_data["Resolution"] == "Won\'t Fix", "Resolution_int"] = 0

# writer = pd.ExcelWriter('/vagrant/examples/data/ALMsINCs-iteration1-all-features.xlsx', engine='xlsxwriter')
# alm_data.to_excel(writer)
# writer.save()

labels = alm_data['Resolution_int']
features = alm_data
features.drop(['Resolution_int', 'Resolution'], 1, inplace=True)

tprs = []
aucs = []
accuracies = []
base_fpr = np.linspace(0, 1, 101)
plt.figure(figsize=(5, 5))

kf = KFold(n_splits=10)
i =0

# Important: otherwise the projection on the index train and test will fill
# the indizes that were removed because the resolution was e.g. 'Cannot reproduce'
# with nan
features = features.reset_index(drop=True)
labels = labels.reset_index(drop=True)


for train, test in kf.split(features):
    print('KFold Iteration: ', i)
    clf = MultinomialNB()
    clf.fit(features.loc[train], labels.loc[train])
    preds = clf.predict_proba(features.loc[test])
    accuracy = clf.score(features.loc[test], labels.loc[test])

    auc = roc_auc_score(labels.loc[test], preds[:, 1])
    fpr, tpr, _ = roc_curve(labels.loc[test], preds[:, 1])

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