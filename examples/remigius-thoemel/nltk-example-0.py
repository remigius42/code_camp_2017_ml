#!/usr/bin/python3
# -*- coding: utf-8 -*-


# ***************************
# Test stop words for german
# ***************************

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob_de import TextBlobDE as TextBlob
from textblob_de import WordDE as Word

def print_as_utf8(word_list):
    print([w.encode('utf-8') for w in list(word_list)])


example_alm = "HPQC-851: Tabelle: Merkmal Spezifikation: Wenn Merkmalgruppen definiert sind sollen nur die Merkmale aus der PS bei den Merkmalgruppen zur Verfügung stehen."

stop_words_de = set(stopwords.words('german'))
stop_words_en = set(stopwords.words('english'))

print(example_alm.encode('utf-8'))
print(len(stop_words_de))
print(len(stop_words_en))
#print_as_utf8(stop_words_de)

word_tokens_alm = word_tokenize(example_alm)

filtered_sentence_alm = [w.encode('utf-8') for w in word_tokens_alm if not w in stop_words_de]
#filtered_sentence_alm = [w for w in word_tokens_alm if not w in stop_words_de]

print(filtered_sentence_alm)


# ***************************
# Test stemming
# ***************************

from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()
snowball = SnowballStemmer("german")

example_words_for_stemmer = ["definiert", "definieren", "definierte", "definition", "definitionen", "definitiv"]
print("PorterStemmer", [ps.stem(w) for w in example_words_for_stemmer])
print("SnowballStemmer", [snowball.stem(w) for w in example_words_for_stemmer])


print("PorterStemmer", [ps.stem(w).encode('utf-8') for w in word_tokens_alm])
print("SnowballStemmer", [snowball.stem(w).encode('utf-8') for w in word_tokens_alm])


# ***************************
# Test lemmatizing
# ***************************
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print("WordNetLemmatizer", [lemmatizer.lemmatize(w) for w in example_words_for_stemmer])
