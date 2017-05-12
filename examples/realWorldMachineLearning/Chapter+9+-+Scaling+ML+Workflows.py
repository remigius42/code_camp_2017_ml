
# coding: utf-8

# # Chapter 9 - Scaling ML Workflows

# In[2]:

get_ipython().magic(u'pylab inline')


# ### Polynomial features

# In[5]:

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import cross_val_score

iris = datasets.load_iris()

linear_classifier = LogisticRegression()
linear_scores = cross_val_score(linear_classifier, iris.data, iris.target, cv=10) #2
print "Accuracy (linear):\t%0.2f (+/- %0.2f)" % (linear_scores.mean(), linear_scores.std() * 2)


pol = PolynomialFeatures(degree=2)
nonlinear_data = pol.fit_transform(iris.data)

nonlinear_classifier = LogisticRegression()
nonlinear_scores = cross_val_score(nonlinear_classifier, nonlinear_data, iris.target, cv=10)
print "Accuracy (nonlinear):\t%0.2f (+/- %0.2f)" % (nonlinear_scores.mean(), nonlinear_scores.std() * 2)

