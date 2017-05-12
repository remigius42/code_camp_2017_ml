
# coding: utf-8

# In[3]:

get_ipython().magic(u'pylab inline')


# # Chapter 4 - Evaluation and Optimization

# We generate two inputs:
# * features – a matrix of input features
# * target – an array of target variables corresponding to those features

# In[4]:

features = rand(100,5)
target = rand(100) > 0.5

print rand
print features
print target


# ### The holdout method
# 
# We divide into a randomized training and test set:

# In[30]:

N = features.shape[0]
N_train = floor(0.7 * N)

print N

# Randomize index
# Note: sometimes you want to retain the order in the dataset and skip this step
# E.g. in the case of time-based datasets where you want to test on 'later' instances

idx = random.permutation(N)
print idx


# In[35]:

# Split index
idx_train = idx[:N_train]
idx_test = idx[N_train:]

print idx_train
print type(idx_train)
print type(features)


# In[3]:

# Break your data into training and testing subsets
features_train = features[idx_train,:]
target_train = target[idx_train]
features_test = features[idx_test,:]
target_test = target[idx_test]

# Build, predict, evaluate (to be filled out)
# model = train(features_train, target_train)
# preds_test = predict(model, features_test)
# accuracy = evaluate_acc(preds_test, target_test)


# In[4]:

print features_train.shape
print features_test.shape
print target_train.shape
print target_test.shape


# ### K-fold cross-validation

# In[5]:

N = features.shape[0]
K = 10 # number of folds

preds_kfold = np.empty(N)
folds = np.random.randint(0, K, size=N)

print 'folds', folds, len(folds)

for idx in np.arange(K):

    # For each fold, break your data into training and testing subsets
    features_train = features[folds != idx,:]
    target_train = target[folds != idx]
    features_test = features[folds == idx,:]
    
    # Print the indices in each fold, for inspection
    print nonzero(folds == idx)[0]

    # Build and predict for CV fold (to be filled out)
    # model = train(features_train, target_train)
    # preds_kfold[folds == idx] = predict(model, features_test)
    
# accuracy = evaluate_acc(preds_kfold, target)


# ### The ROC curve

# In[6]:

def roc_curve(true_labels, predicted_probs, n_points=100, pos_class=1):
    thr = linspace(0,1,n_points)
    tpr = zeros(n_points)
    fpr = zeros(n_points)

    pos = true_labels == pos_class
    neg = logical_not(pos)
    n_pos = count_nonzero(pos)
    n_neg = count_nonzero(neg)
      
    for i,t in enumerate(thr):
        tpr[i] = count_nonzero(logical_and(predicted_probs >= t, pos)) / n_pos
        fpr[i] = count_nonzero(logical_and(predicted_probs >= t, neg)) / n_neg
    
    return fpr, tpr, thr


# In[7]:

# Randomly generated predictions should give us a diagonal ROC curve
preds = rand(len(target))
fpr, tpr, thr = roc_curve(target, preds, pos_class=True)
plot(fpr, tpr)


# ### The area under the ROC curve

# In[8]:

def auc(true_labels, predicted_labels, pos_class=1):
    fpr, tpr, thr = roc_curve(true_labels, predicted_labels,
 pos_class=pos_class)
    area = -trapz(tpr, x=fpr)
    return area


# In[9]:

auc(target, preds, pos_class=True)


# ### Multi-class classification

# In[10]:

d = pandas.read_csv("data/mnist_small.csv")
d_train = d[:int(0.8*len(d))]
d_test = d[int(0.8*len(d)):]


# In[11]:

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(d_train.drop('label', axis=1), d_train['label'])


# In[12]:

from sklearn.metrics import confusion_matrix
preds = rf.predict(d_test.drop('label', axis=1))
cm = confusion_matrix(d_test['label'], preds)


# In[13]:

matshow(cm, cmap='Greys')
colorbar()
savefig("figures/figure-4.19.eps", format='eps')


# ### The root-mean-square error

# In[14]:

def rmse(true_values, predicted_values):
    n = len(true_values)
    residuals = 0
    for i in range(n):
        residuals += (true_values[i] - predicted_values[i])**2.
    return np.sqrt(residuals/n)


# In[15]:

rmse(rand(10), rand(10))


# ### The R-squared error

# In[16]:

def r2(true_values, predicted_values):
    n = len(true_values)
    mean = np.mean(true_values)
    residuals = 0
    total = 0
    for i in range(n):
        residuals += (true_values[i] - predicted_values[i])**2.
        total += (true_values[i] - mean)**2.
    return 1.0 - residuals/total


# In[17]:

r2(arange(10)+rand(), arange(10)+rand(10))


# ### Grid search with kernel-SVM model
# 
# Importing modules:

# In[18]:

from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC


# Loading data and performang poor-mans feature engineering:

# In[19]:

d = pandas.read_csv("data/titanic.csv")

# Target
y = d["Survived"]

# Features
X = d.drop(["Survived", "PassengerId", "Cabin","Ticket","Name", "Fare"], axis=1)
X['Sex'] = map(lambda x: 1 if x=="male" else 0, X['Sex'])
X['Embarked-Q'] = map(lambda x: 1 if x=="Q" else 0, X['Embarked'])
X['Embarked-C'] = map(lambda x: 1 if x=="C" else 0, X['Embarked'])
X['Embarked-S'] = map(lambda x: 1 if x=="S" else 0, X['Embarked'])
X = X.drop(["Embarked", "Sex"], axis=1)
X = X.fillna(-1)


# Performing grid-search to find the optimal hyper-parameters:

# In[20]:

# grid of (gamma, C) values to try 
gam_vec, cost_vec = np.meshgrid(np.linspace(0.01, 10, 11),
                     np.linspace(0.01, 10, 11))

AUC_all = [] # initialize empty array to store AUC results

# set up cross-validation folds
N = len(y)
K = 10 # number of cross-validation folds
folds = np.random.randint(0, K, size=N)

# search over every value of the grid
for param_ind in np.arange(len(gam_vec.ravel())):

    # initialize cross-validation predictions
    y_cv_pred = np.empty(N)

    # loop through the cross-validation folds
    for ii in np.arange(K):
        # break your data into training and testing subsets
        X_train = X.ix[folds != ii,:]
        y_train = y.ix[folds != ii]
        X_test = X.ix[folds == ii,:]

        # build a model on the training set
        model = SVC(gamma=gam_vec.ravel()[param_ind], C=cost_vec.ravel()[param_ind])
        model.fit(X_train, y_train)

        # generate and store model predictions on the testing set
        y_cv_pred[folds == ii] = model.predict(X_test)

    # evaluate the AUC of the predictions
    AUC_all.append(roc_auc_score(y, y_cv_pred))

indmax = np.argmax(AUC_all)
print "Maximum = %.3f" % (np.max(AUC_all))
print "Tuning Parameters: (gamma = %.2f, C = %.2f)" % (gam_vec.ravel()[indmax], cost_vec.ravel()[indmax])


# Plotting the contours of the parameter performance:

# In[21]:

AUC_grid = np.array(AUC_all).reshape(gam_vec.shape)

contourf(gam_vec, cost_vec, AUC_grid, 20, cmap='Greys')
xlabel("kernel coefficient, gamma")
ylabel("penalty parameter, C")
colorbar()
savefig("figures/figure-4.25.eps", format='eps')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



