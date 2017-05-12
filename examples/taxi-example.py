import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score




# data collected from here: http://odysseus.informatik.uni-oldenburg.de/download/Data/Grand%20Challenge/2015/original/
# Read the data into pandas
N = 5e6
data = pd.read_csv("/vagrant/examples/data/trip_data_1_100000.csv", nrows=N)
fare_data = pd.read_csv("/vagrant/examples/data/trip_fare_1_100000.csv", nrows=N)

fare_cols = [u' payment_type', u' fare_amount', u' surcharge', u' mta_tax', u' tip_amount', u' tolls_amount', u' total_amount']

# merge them into a single file
data = data.join(fare_data[fare_cols])

# print the first records
print('First rows of merged data', data.ix[:5, data.columns[:5]])
#print(data[:10])

# Inspect and visualize the features
# scatter for numeric values
data.plot(x="trip_time_in_secs", y=" total_amount", kind="scatter", s=2)
plt.show()

# bar histogram for categorical values
data.rate_code.value_counts().plot(kind="bar", logy=True, ylim=(1,1e8))
plt.show()



# preprocess the label
# make it numeric with values 0 and 1
data['tipped'] = (data[' tip_amount'] > 0).astype("int")
print('Distribution of label column: ', data['tipped'].value_counts())

# choose the features to be considered
feats1 = [u'rate_code', 'passenger_count', u'trip_time_in_secs', u'trip_distance', u'pickup_longitude', u'pickup_latitude', u'dropoff_longitude', u'dropoff_latitude', ' fare_amount', u' surcharge', u' mta_tax', ' tolls_amount']


# split into training and test ids
# make an array with the indizes from 0 to len(data)
M = len(data)
rand_idx = np.arange(M)
# shuffle them
np.random.shuffle(rand_idx)
# partition the indizes
train_idx = rand_idx[int(M*0.2):]
test_idx = rand_idx[:int(M*0.2)]


# scale all features into range 0..1
# to make sure their weight is equal
sc = StandardScaler()
data_scaled = sc.fit_transform(data[feats1])
print('Shape of scaled data: ', data_scaled[train_idx.tolist(),:].shape)

# create a linera classifier
sgd = SGDClassifier(loss="modified_huber")
# see http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.ix.html
# choose data found in defined index array
sgd.fit(data.ix[train_idx, feats1], data['tipped'].ix[train_idx])
print('Linear classifier: ', sgd)

# predict with propabilities, run on test data
preds = sgd.predict_proba(data.ix[test_idx,feats1])

# calculate roc and auc
fpr, tpr, thr = roc_curve(data['tipped'].ix[test_idx], preds[:,1])
auc = roc_auc_score(data['tipped'].ix[test_idx], preds[:,1])
print('AUC for linear classifier: ', auc)

# and plot it
plt.plot(fpr,tpr)
plt.plot(fpr,fpr)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()




# ----------------
# Random Forest
# ----------------

# null values are bad for decision trees -> fill them with 0
data.fillna(0, inplace=True)
#print('Nonzero :', np.count_nonzero(pd.isnull(data.ix[train_idx,feats1])))

# take random forest model
rf1 = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf1.fit(data.ix[train_idx,feats1], data['tipped'].ix[train_idx])
print('Random Forest classifier: ', rf1)

# calculate predictions with probabilities
preds1 = rf1.predict_proba(data.ix[test_idx,feats1])

fpr1, tpr1, thr1 = roc_curve(data['tipped'].ix[test_idx], preds1[:,1])
auc1 = roc_auc_score(data['tipped'].ix[test_idx], preds1[:,1])
print(' AUC for RandomForest: ', auc1)

# calculate mean
print('Mean for test data: ', rf1.score(data.ix[test_idx,feats1], data.ix[test_idx,'tipped']))

plt.plot(fpr1,tpr1)
plt.plot(fpr1,fpr1)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()

# ---------------
# Optimize features
# ---------------


fi = list(zip(feats1, rf1.feature_importances_))
fi.sort(key=lambda x: -x[1])
print('Feature Importance :', pd.DataFrame(fi, columns=["Feature","Importance"]))

data['trip_time_in_secs'][data['trip_time_in_secs'] < 1e-3] = -1
data['speed'] = data['trip_distance'] / data['trip_time_in_secs']

feats2 = feats1 + ['speed']
feats2.remove('trip_time_in_secs')

rf2 = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf2.fit(data.ix[train_idx,feats2], data['tipped'].ix[train_idx])

preds2 = rf2.predict_proba(data.ix[test_idx,feats2])

fpr2, tpr2, thr2 = roc_curve(data['tipped'].ix[test_idx], preds2[:,1])
auc2 = roc_auc_score(data['tipped'].ix[test_idx], preds2[:,1])

print('AUC with new feature :', auc2)

plt.plot(fpr1,tpr1)
plt.plot(fpr1,fpr1)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()