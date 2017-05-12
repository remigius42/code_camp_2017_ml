
# coding: utf-8

# # Chapter 7 - Advanced Feature Engineering

# In[1]:

get_ipython().magic(u'pylab inline')


# ### Latent Semantic Analysis

# In[2]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def latent_semantic_analysis(docs):
  tfidf = TfidfVectorizer() # Using default parameters
  tfidf.fit(docs) # Creating dictionary
  vecs = tfidf.transform(docs) # Using dictionary to vectorize documents
  svd = TruncatedSVD(n_components=100) # Generating 100 top components
  svd.fit(vecs) # Creating SVD matrices
  return svd.transform(vecs) # Finally use LSA to vectorize documents


# In[12]:

from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')
latent_semantic_analysis(newsgroups_train.data)


# ### Latent Dirichlet Analysis

# In[14]:

from gensim.models.ldamodel import LdaModel

def lda_model(docs):
  # Build LDA model, setting the number of topics to extract
  return LdaModel(docs, num_topics=20)

def lda_vector(lda_model, doc):
  # Generate features for a new document
  return lda_model[doc]


# In[17]:

from gensim.utils import mock_data
#gensim_docs = [d.split(" ") for d in newsgroups_train.data]
gensim_corpus = mock_data()
lda = lda_model(gensim_corpus)


# ### Histogram of Oriented Gradients 
# 
# See the full example in the Scikit-Image Gallery:
# 
# http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html

# In[25]:

from skimage import data, color, feature

image = color.rgb2gray(data.lena())
feature.hog(image, orientations=9, pixels_per_cell=(8,8),
      cells_per_block=(3,3), normalise=True, visualise=True)


# ### Event data to time-series

# In[28]:

import pandas as pd
from datetime import datetime

df = pd.read_csv("data/sfpd_incident_all.csv")

df['Month'] = map(lambda x: datetime.strptime("/".join(x.split("/")[0::2]),"%m/%Y"),df['Date'])

# create classical time series
df_ts = df.groupby('Month').aggregate(len)["IncidntNum"]

# plot time series
plot(df_ts.index,df_ts.values,'-k',lw=2)
xlabel("Month")
ylabel("Number of Crimes")
ylim((8000, 14000))


# ### Windowed Statistics

# In[30]:

# window = spring 2014
window1 = (datetime(2014,3,22),datetime(2014,6,21))

# find which data points fall within the window
idx_window = np.where(map(lambda x: x>=window1[0] and x<=window1[1], df_ts.index))[0]

# windowed mean and standard deviation
mean_window = np.mean(df_ts.values[idx_window])
std_window = np.std(df_ts.values[idx_window])

# windowed differences:
# window 2 = spring 2013
window2 = (datetime(2013,3,22),datetime(2013,6,21))

# find which data points fall within the window
idx_window2 = np.where(map(lambda x: x>=window2[0] and x<=window2[1], df_ts.index))[0]

# windowed differences: mean and standard deviation
mean_wdiff = mean_window - np.mean(df_ts.values[idx_window2])
std_wdiff = std_window - np.std(df_ts.values[idx_window2])


# ### Periodogram features

# In[31]:

import scipy.signal

# compute the periodogram
f, psd = scipy.signal.periodogram(df_ts, detrend='linear')
plt.plot(f, psd,'-ob')
plt.xlabel('frequency [1/month]')
plt.ylabel('Spectral Density')
plt.show()

# Features:
# period of highest psd peak:
period_psd1 = 1./f[np.argmax(psd)] # = 47.0 months

# sum of spectral density higher than 1/12 months
sdens_gt_12m = np.sum(psd[f > 1./12])
# ratio of spectral density higher than to less than 1/12 months
sdens_ratio_12m = float(sdens_gt_12m) / np.sum(psd[f <= 1./12])

