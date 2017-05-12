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


np.random.seed(0)

feature_columns = ['Summary', 'Issue Type', 'Priority', 'Reporter', 'Creator', 'Created', 'Affects Version/s', 'Images', 'Environment', 'Description', 'Resolution']
lower_bound_w_count = 5
upper_bound_w_count = 100
min_word_len = 4
summary_lex_file = '/vagrant/examples/data/summary_lex_iteration_0.pickle'
description_lex_file = '/vagrant/examples/data/description_lex_iteration_0.pickle'

alm_data = pd.read_excel('/vagrant/examples/data/ALMsINCs.xlsx')
#alm_data[feature_columns].to_csv('/vagrant/examples/data/ALMforWeka.csv',index=False)

stop_words_de = set(stopwords.words('german'))
#stemmer = SnowballStemmer("german")
stemmer = PorterStemmer()


def cat_to_binary(col_name, data):
    col_data = data[col_name]
    categories = np.unique(col_data)
    for cat in categories:
        binary = (col_data == cat)
        new_column_name = col_name + '-' + str(cat)
        data[new_column_name] = binary.astype("int")
    return


print(alm_data.shape)
alm_data = alm_data[((alm_data['Resolution'] == 'Fixed') | (alm_data['Resolution'] == 'Won\'t Fix'))]
print(alm_data.shape)
alm_data = alm_data[feature_columns]
print(alm_data.shape)

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





def build_lexicon_for_column(column_name):
    lexicon = []
    word_tokens = []
    for index, row in alm_data.iterrows():
        # print(row[column_name])
        word_tokens = word_tokenize(str(row[column_name]))
        lexicon += list(word_tokens)

    print('number of words in column ', column_name, ': ', len(lexicon))
#    print(lexicon[:100])

    lexicon = [stemmer.stem(str(w)) for w in lexicon]
    w_counts = Counter(lexicon)
    print('number of distinct words: ', len(w_counts))
 #   print(list(w_counts)[:100])


    filtered_w_counts = [w for w in w_counts if not w in stop_words_de]
    print('number of filtered distinct words: ', len(filtered_w_counts))
#    print(filtered_w_counts[:100])

    lexicon_with_relevant_words = []

    for word in filtered_w_counts:
#        print(word, "': ", w_counts[word])
        if upper_bound_w_count > w_counts[word] > lower_bound_w_count:
            if len(word) >= min_word_len:
                lexicon_with_relevant_words.append(word)

#    print('number of relevant words in lexicon: ', len(lexicon_with_relevant_words))
#    print(lexicon_with_relevant_words[:100])
    return lexicon_with_relevant_words




if os.path.exists(summary_lex_file):
    f = open(summary_lex_file, 'rb')
    summary_lex = pickle.load(f)
    f.close()
else:
    summary_lex = build_lexicon_for_column('Summary')
    with open(summary_lex_file,'wb') as f:
        pickle.dump(summary_lex, f, pickle.HIGHEST_PROTOCOL)


if os.path.exists(description_lex_file):
    f = open(description_lex_file, 'rb')
    description_lex = pickle.load(f)
    f.close()
else:
    description_lex = build_lexicon_for_column('Description')
    with open(description_lex_file,'wb') as f:
        pickle.dump(description_lex, f, pickle.HIGHEST_PROTOCOL)


def create_summary_column_name(word):
    return 'summary_' + str(word)


def create_description_column_name(word):
    return 'description_' + str(word)


summary_columns = []
description_columns = []

def add_columns_for_word_dictionary(df, word_dict):
    columns = []
    for word in word_dict:
        df[create_description_column_name(word)] = 0
        columns.append(create_description_column_name(word))
    print("New shape:", alm_data.shape)
    return df, columns

_, summary_columns = add_columns_for_word_dictionary(alm_data, summary_lex)
_, description_columns = add_columns_for_word_dictionary(alm_data, description_lex)

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

print(alm_data.head())


# Random forest
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()


# *********************************
#  Evaluate Model
# *********************************

labels = alm_data['Resolution']
features = alm_data
features.drop(['Resolution'], 1, inplace=True)



X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=0)

clf.fit(X_train, np.ravel(y_train))

# predict with propabilities, run on test data
preds = clf.predict_proba(X_test)
accuracy = clf.score(X_test, y_test)
print(accuracy)

def export_test_data():
    alm_data = pd.read_excel('/vagrant/examples/data/ALMsINCs.xlsx')
    alm_data = alm_data[feature_columns]
    verification = X_test.copy()
    verification['Test_Class'] = clf.predict(X_test)
    pred_probs = clf.predict_proba(X_test)
    verification['Test_Prob_Fixed'] = pred_probs[:,0]
    verification['Test_Prob_Wont'] = pred_probs[:,1]
    verification = verification[['Test_Class', 'Test_Prob_Fixed', 'Test_Prob_Wont']]
    alm_data = alm_data.loc[verification.index]
    alm_data['Test_Class'] = verification['Test_Class']
    alm_data['Test_Prob_Fixed'] = verification['Test_Prob_Fixed']
    alm_data['Test_Prob_Wont'] = verification['Test_Prob_Wont']

    writer = pd.ExcelWriter('/vagrant/examples/data/ALMsINCsIteration0.xlsx', engine='xlsxwriter')
    alm_data.to_excel(writer)
    writer.save()
    
export_test_data()
