
# coding: utf-8

# # Chapter 8 - Movie Review Example

# In[1]:

get_ipython().magic(u'pylab inline')


# In[2]:

import pandas


# In[3]:

d = pandas.read_csv("data/movie_reviews.tsv", delimiter="\t")


# In[4]:

# Holdout split
split = 0.7
d_train = d[:int(split*len(d))]
d_test = d[int((1-split)*len(d)):]


# In[5]:

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(d_train.review)


# In[6]:

i = 45000
j = 10
words = vectorizer.get_feature_names()[i:i+10]
pandas.DataFrame(features[j:j+7,i:i+10].todense(), columns=words)


# In[7]:

float(features.getnnz())*100 / (features.shape[0]*features.shape[1])


# In[8]:

from sklearn.naive_bayes import MultinomialNB
model1 = MultinomialNB()
model1.fit(features, d_train.sentiment)


# In[9]:

pred1 = model1.predict_proba(vectorizer.transform(d_test.review))


# In[10]:

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
def performance(y_true, pred, color="g", ann=True):
    acc = accuracy_score(y_true, pred[:,1] > 0.5)
    auc = roc_auc_score(y_true, pred[:,1])
    fpr, tpr, thr = roc_curve(y_true, pred[:,1])
    plot(fpr, tpr, color, linewidth="3")
    xlabel("False positive rate")
    ylabel("True positive rate")
    if ann:
        annotate("Acc: %0.2f" % acc, (0.1,0.8), size=14)
        annotate("AUC: %0.2f" % auc, (0.1,0.7), size=14)


# In[11]:

performance(d_test.sentiment, pred1)


# ## tf-idf features

# In[21]:

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(d_train.review)


# In[22]:

pred2 = model1.predict_proba(vectorizer.transform(d_test.review))
performance(d_test.sentiment, pred1, ann=False)
performance(d_test.sentiment, pred2, color="b")
xlim(0,0.5)
ylim(0.5,1)


# ## Parameter optimization

# In[23]:

param_ranges = {
    "max_features": [10000, 30000, 50000, None],
    "min_df": [1,2,3],
    "nb_alpha": [0.01, 0.1, 1.0]
}


# In[24]:

def build_model(max_features=None, min_df=1, nb_alpha=1.0, return_preds=False):
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df)
    features = vectorizer.fit_transform(d_train.review)
    model = MultinomialNB(alpha=nb_alpha)
    model.fit(features, d_train.sentiment)
    pred = model.predict_proba(vectorizer.transform(d_test.review))
    res = {
        "max_features": max_features,
        "min_df": min_df,
        "nb_alpha": nb_alpha,
        "auc": roc_auc_score(d_test.sentiment, pred[:,1])
    }
    if return_preds:
        res['preds'] = pred
    return res


# In[25]:

from itertools import product
results = []
for p in product(*param_ranges.values()): 
    res = build_model(**dict(zip(param_ranges.keys(), p)))
    results.append( res )
    print res


# In[26]:

opt = pandas.DataFrame(results)


# In[27]:

mf_idx = [0,9,18,27]
plot(opt.max_features[mf_idx], opt.auc[mf_idx], linewidth=2)
title("AUC vs max_features")


# In[28]:

mdf_idx = [27,28,29]
plot(opt.min_df[mdf_idx], opt.auc[mdf_idx], linewidth=2)
title("AUC vs min_df")


# In[29]:

nba_idx = [27,30,33]
plot(opt.nb_alpha[nba_idx], opt.auc[nba_idx], linewidth=2)
title("AUC vs alpha")


# In[30]:

pred3 = build_model(nb_alpha=0.01, return_preds=True)['preds']
performance(d_test.sentiment, pred1, ann=False)
performance(d_test.sentiment, pred2, color="b", ann=False)
performance(d_test.sentiment, pred3, color="r")
xlim(0,0.5)
ylim(0.5,1)


# ## Random Forest

# In[31]:

vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words='english', min_df=3, max_features=30000, norm="l2")
features = vectorizer.fit_transform(d_train.review)


# In[32]:

model3 = MultinomialNB()
model3.fit(features, d_train.sentiment)
pred3 = model3.predict_proba(vectorizer.transform(d_test.review))
performance(d_test.sentiment, pred3)


# In[33]:

from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(n_estimators=100)
model2.fit(features, d_train.sentiment)


# In[28]:

pred2 = model2.predict_proba(vectorizer.transform(d_test.review))
performance(d_test.sentiment, pred2)


# ## Word2Vec

# In[29]:

import re, string
stop_words = set(['all', "she'll", "don't", 'being', 'over', 'through', 'yourselves', 'its', 'before', "he's", "when's", "we've", 'had', 'should', "he'd", 'to', 'only', "there's", 'those', 'under', 'ours', 'has', "haven't", 'do', 'them', 'his', "they'll", 'very', "who's", "they'd", 'cannot', "you've", 'they', 'not', 'during', 'yourself', 'him', 'nor', "we'll", 'did', "they've", 'this', 'she', 'each', "won't", 'where', "mustn't", "isn't", "i'll", "why's", 'because', "you'd", 'doing', 'some', 'up', 'are', 'further', 'ourselves', 'out', 'what', 'for', 'while', "wasn't", 'does', "shouldn't", 'above', 'between', 'be', 'we', 'who', "you're", 'were', 'here', 'hers', "aren't", 'by', 'both', 'about', 'would', 'of', 'could', 'against', "i'd", "weren't", "i'm", 'or', "can't", 'own', 'into', 'whom', 'down', "hadn't", "couldn't", 'your', "doesn't", 'from', "how's", 'her', 'their', "it's", 'there', 'been', 'why', 'few', 'too', 'themselves', 'was', 'until', 'more', 'himself', "where's", "i've", 'with', "didn't", "what's", 'but', 'herself', 'than', "here's", 'he', 'me', "they're", 'myself', 'these', "hasn't", 'below', 'ought', 'theirs', 'my', "wouldn't", "we'd", 'and', 'then', 'is', 'am', 'it', 'an', 'as', 'itself', 'at', 'have', 'in', 'any', 'if', 'again', 'no', 'that', 'when', 'same', 'how', 'other', 'which', 'you', "shan't", 'our', 'after', "let's", 'most', 'such', 'on', "he'll", 'a', 'off', 'i', "she'd", 'yours', "you'll", 'so', "we're", "she's", 'the', "that's", 'having', 'once'])

def tokenize(docs):
    pattern = re.compile('[\W_]+', re.UNICODE)
    sentences = []
    for d in docs:
        sentence = d.lower().split(" ") 
        sentence = [pattern.sub('', w) for w in sentence]
        sentences.append( [w for w in sentence if w not in stop_words] )
    return sentences


# In[30]:

print list(stop_words)


# In[31]:

sentences = tokenize(d_train.review)


# In[33]:

from gensim.models.word2vec import Word2Vec
model = Word2Vec(sentences, size=300, window=10, min_count=1, sample=1e-3, workers=2)


# In[34]:

model.init_sims(replace=True)


# In[35]:

model['movie']


# In[36]:

def featurize_w2v(model, sentences):
    f = zeros((len(sentences), model.vector_size))
    for i,s in enumerate(sentences):
        for w in s:
            try:
                vec = model[w]
            except KeyError:
                continue
            f[i,:] = f[i,:] + vec
        f[i,:] = f[i,:] / len(s)
    return f


# In[37]:

features_w2v = featurize_w2v(model, sentences)


# In[38]:

model4 = RandomForestClassifier(n_estimators=100, n_jobs=-1)
model4.fit(features_w2v, d_train.sentiment)


# In[39]:

test_sentences = tokenize(d_test.review)


# In[40]:

test_features_w2v = featurize_w2v(model, test_sentences)


# In[41]:

pred4 = model4.predict_proba(test_features_w2v)


# In[42]:

performance(d_test.sentiment, pred1, ann=False)
performance(d_test.sentiment, pred2, color="b", ann=False)
performance(d_test.sentiment, pred3, color="r", ann=False)
performance(d_test.sentiment, pred4, color="c")
xlim(0,0.3)
ylim(0.6,1)


# In[48]:

examples = [
        "This movie is bad",
        "This movie is great",
        "I was going to say something awesome, but I simply can't because the movie is so bad.",
        "I was going to say something awesome or great or good, but I simply can't because the movie is so bad.",
        "It might have bad actors, but everything else is good."
    ]
example_feat4 = featurize_w2v(model, tokenize(examples))
model4.predict(example_feat4)

