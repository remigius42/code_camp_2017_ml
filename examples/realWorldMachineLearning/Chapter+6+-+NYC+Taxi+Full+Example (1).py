
# coding: utf-8

# # Chapter 6 - NYC Taxi Full Example
# 
# http://chriswhong.com/open-data/foil_nyc_taxi/
# 
# http://www.andresmh.com/nyctaxitrips/
# 
# Potential models
# 
# * Will the passenger tip the driver?
# * Will the trip take longer than average?
# * Will the fare be higher than average?

# In[3]:

import pandas
get_ipython().magic(u'pylab inline')


# ## Data exploration

# In[4]:

N = 5e6
data = pandas.read_csv("trip_data_1_100000.csv", nrows=N)


# In[5]:

fare_data = pandas.read_csv("trip_fare_1_100000.csv", nrows=N)
fare_cols = [u' payment_type', u' fare_amount', u' surcharge', u' mta_tax', u' tip_amount', u' tolls_amount', u' total_amount']
data = data.join(fare_data[fare_cols])
del fare_data
print data[:10]


# In[6]:

data.ix[:5, data.columns[:5]]


# In[7]:

data.plot(x="trip_time_in_secs", y=" total_amount", kind="scatter", s=2)
xlim(0,1e4)
ylim(0,300)


# In[8]:

ind = where(logical_and(data.trip_time_in_secs < 500, data[' total_amount'] > 30))[0]
data = data.drop(ind)


# In[9]:

data[logical_and(data.dropoff_latitude > 40.6,data.dropoff_latitude < 40.9)].dropoff_latitude.hist(bins=20);


# In[10]:

data[logical_and(data.dropoff_longitude > -74.05,data.dropoff_longitude < -73.9)].dropoff_longitude.hist(bins=20);


# In[11]:

data.vendor_id.value_counts().plot(kind="bar");


# In[12]:

data.rate_code.value_counts().plot(kind="bar", logy=True, ylim=(1,1e8));


# In[13]:

data.store_and_fwd_flag.value_counts().plot(kind="bar");


# In[14]:

data.passenger_count.value_counts().plot(kind="bar");


# In[15]:

data.trip_time_in_secs[data.trip_time_in_secs < 4000].hist(bins=30);


# In[16]:

data.trip_distance[data.trip_distance < 22].hist(bins=30);


# In[17]:

data[' payment_type'].value_counts().plot(kind="bar", logy=True, ylim=(1,1e8));


# In[18]:

data.plot(x="trip_time_in_secs", y="trip_distance", kind="scatter", s=2)
xlim(0,5000)
ylim(0,40)


# In[19]:

figure(figsize=(16,8))
plot(data["pickup_latitude"], data["pickup_longitude"], 'b,')
xlim(40.6, 40.9)
ylim(-74.05, -73.9)


# In[20]:

data[data[' tip_amount'] < 15][' tip_amount'].hist(bins=30);


# In[21]:

len(data)
data = data[data[' payment_type'] != "CSH"]
data.reset_index(inplace=True, drop=True)
len(data)


# ## Building first model

# In[22]:

# Setup target
data['tipped'] = (data[' tip_amount'] > 0).astype("int")
data['tipped'].value_counts()


# In[23]:

feats1 = [u'rate_code', 'passenger_count', u'trip_time_in_secs', u'trip_distance', u'pickup_longitude', u'pickup_latitude', u'dropoff_longitude', u'dropoff_latitude', ' fare_amount', u' surcharge', u' mta_tax', ' tolls_amount']


# In[24]:

M = len(data)
rand_idx = arange(M)
random.shuffle(rand_idx)
train_idx = rand_idx[int(M*0.2):]
test_idx = rand_idx[:int(M*0.2)]


# In[26]:

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[27]:

sc = StandardScaler()
data_scaled = sc.fit_transform(data[feats1])
data_scaled[train_idx.tolist(),:].shape


# In[28]:

sgd = SGDClassifier(loss="modified_huber")
sgd.fit(data.ix[train_idx,feats1], data['tipped'].ix[train_idx])


# In[36]:

preds = sgd.predict_proba(data.ix[test_idx,feats1])


# In[37]:

fpr, tpr, thr = roc_curve(data['tipped'].ix[test_idx], preds[:,1])
auc = roc_auc_score(data['tipped'].ix[test_idx], preds[:,1])


# In[38]:

auc


# In[39]:

plot(fpr,tpr)
plot(fpr,fpr)
xlabel("False positive rate")
ylabel("True positive rate")


# In[29]:

## Random Forest


# In[30]:

from sklearn.ensemble import RandomForestClassifier


# In[33]:

data.fillna(0, inplace=True)
count_nonzero(pandas.isnull(data.ix[train_idx,feats1]))


# In[34]:

rf1 = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf1.fit(data.ix[train_idx,feats1], data['tipped'].ix[train_idx])


# In[35]:

preds1 = rf1.predict_proba(data.ix[test_idx,feats1])


# In[36]:

from sklearn.metrics import roc_curve, roc_auc_score


# In[37]:

fpr1, tpr1, thr1 = roc_curve(data['tipped'].ix[test_idx], preds1[:,1])
auc1 = roc_auc_score(data['tipped'].ix[test_idx], preds1[:,1])


# In[38]:

print auc1
rf1.score(data.ix[test_idx,feats1], data.ix[test_idx,'tipped'])


# In[47]:

plot(fpr1,tpr1)
plot(fpr1,fpr1)
xlabel("False positive rate")
ylabel("True positive rate")


# In[54]:

fi = zip(feats1, rf1.feature_importances_)
print(rf1.feature_importances_)
fi.sort(key=lambda x: -x[1])
pandas.DataFrame(fi, columns=["Feature","Importance"])


# ### Features 2

# In[40]:

data['trip_time_in_secs'][data['trip_time_in_secs'] < 1e-3] = -1
data['speed'] = data['trip_distance'] / data['trip_time_in_secs']


# In[41]:

feats2 = feats1 + ['speed']
feats2.remove('trip_time_in_secs')


# In[42]:

rf2 = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf2.fit(data.ix[train_idx,feats2], data['tipped'].ix[train_idx])


# In[43]:

preds2 = rf2.predict_proba(data.ix[test_idx,feats2])


# In[44]:

fpr2, tpr2, thr2 = roc_curve(data['tipped'].ix[test_idx], preds2[:,1])
auc2 = roc_auc_score(data['tipped'].ix[test_idx], preds2[:,1])


# In[45]:

print auc2
plot(fpr2,tpr2)
plot(fpr2,fpr2)


# In[51]:

fi2 = zip(feats2, rf2.feature_importances_)
fi2.sort(key=lambda x: x[1])
fi2


# ### Features 3

# In[39]:

feats3 = feats1


# In[58]:

feats3


# In[59]:

from sklearn.feature_extraction import DictVectorizer


# In[60]:

def cat_to_num(data):
    categories = unique(data)
    features = {}
    for cat in categories:
        binary = (data == cat)
        features["%s:%s"%(data.name, cat)] = binary.astype("int")
    return pandas.DataFrame(features)


# In[61]:

payment_type_cats = cat_to_num(data[' payment_type'])
vendor_id_cats = cat_to_num(data['vendor_id'])
store_and_fwd_flag_cats = cat_to_num(data['store_and_fwd_flag'])
rate_code = cat_to_num(data['rate_code'])


# In[62]:

data = data.join(payment_type_cats)
feats3 += payment_type_cats.columns
data = data.join(vendor_id_cats)
feats3 += vendor_id_cats.columns
data = data.join(store_and_fwd_flag_cats)
feats3 += store_and_fwd_flag_cats.columns
data = data.join(rate_code)
feats3 += rate_code.columns


# In[ ]:

feats3


# In[ ]:

rf3 = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf3.fit(data.ix[train_idx,feats3], data['tipped'].ix[train_idx])


# In[ ]:

rf3.score(data.ix[test_idx,feats3], data.ix[test_idx,'tipped'])


# In[ ]:

fpr3, tpr3, thr3 = roc_curve(data['tipped'].ix[test_idx], preds3[:,1])
auc3 = roc_auc_score(data['tipped'].ix[test_idx], preds3[:,1])
print auc3
plot(fpr3,tpr3)
plot(fpr3,fpr3)
xlabel("False positive rate")
ylabel("True positive rate")


# In[ ]:

fi3 = zip(feats3, rf3.feature_importances_)
fi3.sort(key=lambda x: -x[1])
pandas.DataFrame(fi3, columns=["Feature","Importance"])


# ### Features 4

# In[ ]:

feats4 = feats3


# In[ ]:

# Datetime features (hour of day, day of week, week of year)
pickup = pandas.to_datetime(data['pickup_datetime'])
dropoff = pandas.to_datetime(data['dropoff_datetime'])
data['pickup_hour'] = pickup.apply(lambda e: e.hour)
data['pickup_day'] = pickup.apply(lambda e: e.dayofweek)
#data['pickup_week'] = pickup.apply(lambda e: e.week)
data['dropoff_hour'] = dropoff.apply(lambda e: e.hour)
data['dropoff_day'] = dropoff.apply(lambda e: e.dayofweek)
#data['dropoff_week'] = dropoff.apply(lambda e: e.week)


# In[ ]:

feats4 += ['pickup_hour', 'pickup_day', 'dropoff_hour', 'dropoff_day']


# In[ ]:

feats4


# In[ ]:

rf4 = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf4.fit(data.ix[train_idx,feats4], data['tipped'].ix[train_idx])


# In[ ]:

preds4 = rf4.predict_proba(data.ix[test_idx,feats4])
rf4.score(data.ix[test_idx,feats4], data.ix[test_idx,'tipped'])


# In[ ]:

fpr4, tpr4, thr4 = roc_curve(data['tipped'].ix[test_idx], preds4[:,1])
auc4 = roc_auc_score(data['tipped'].ix[test_idx], preds4[:,1])
print auc4
figure(figsize=(14,8))
plot(fpr4,tpr4, "g-", linewidth=3)
plot(fpr4,fpr4, "k-", linewidth=1)
xlabel("False positive rate")
ylabel("True positive rate")


# In[ ]:

fi4 = zip(feats4, rf4.feature_importances_)
fi4.sort(key=lambda x: -x[1])
pandas.DataFrame(fi4, columns=["Feature","Importance"])


# In[ ]:

data.ix[data[' payment_type:CSH'] == 1,'tipped'].value_counts()


# In[ ]:

figure(figsize=(16,8))
plot(data[data['tipped'] == True]["dropoff_latitude"], data[data['tipped'] == True]["dropoff_longitude"], 'b,')
plot(data[data['tipped'] == False]["dropoff_latitude"], data[data['tipped'] == False]["dropoff_longitude"], 'r,')
xlim(40.6, 40.9)
ylim(-74.05, -73.9)

