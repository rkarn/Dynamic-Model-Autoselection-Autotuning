import os
path='/Users/rupesh.karn/Desktop/InternshipStuffs/Hessian + Deep Learning University Stuffs'
os.chdir(path)

print os.listdir(path)

import pandas as pd
import numpy as np
import csv
import os
import time
from numba import vectorize, cuda
#from matplotlib.pyplot import *
#import matplotlib.pyplot as plt
#import pylab
#from sklearn.cross_validation import train_test_split
#from sklearn.metrics import accuracy_score
#pylab.rcParams['figure.figsize'] = (16.0, 5.0)

# Read in the training CSV file
print("Reading Training csv file.")
df1 = pd.read_csv("UNSW_NB15_training-set.csv")
df1.drop('label', axis=1, inplace=True)

obj_df=df1

obj_df["proto"] = obj_df["proto"].astype('category')
obj_df["service"] = obj_df["service"].astype('category')
obj_df["state"] = obj_df["state"].astype('category')
obj_df["proto_cat"] = obj_df["proto"].cat.codes
obj_df["service_cat"] = obj_df["service"].cat.codes
obj_df["state_cat"] = obj_df["state"].cat.codes

obj_df["proto"] = obj_df["proto_cat"]
obj_df["service"] = obj_df["service_cat"]
obj_df["state"] = obj_df["state_cat"]

obj_df.drop('proto_cat', axis=1, inplace=True)
obj_df.drop('service_cat', axis=1, inplace=True)
obj_df.drop('state_cat', axis=1, inplace=True)

Y_train_all_attacks = obj_df["attack_cat"]
obj_df=pd.get_dummies(obj_df, columns=["attack_cat"])


X_train = obj_df.values[:,:-10]


for j in range(0,43):
    maximum = max(X_train[:,j])
    for i in range(0,len(X_train)):
        X_train[i,j] = round(X_train[i,j]/maximum,3)

# Read in the testing CSV file 
print("Reading Testing csv file.")
df2 = pd.read_csv("UNSW_NB15_testing-set.csv")
df2.drop('label', axis=1, inplace=True)

obj_df2=df2

obj_df2["proto"] = obj_df2["proto"].astype('category')
obj_df2["service"] = obj_df2["service"].astype('category')
obj_df2["state"] = obj_df2["state"].astype('category')
obj_df2["proto_cat"] = obj_df2["proto"].cat.codes
obj_df2["service_cat"] = obj_df2["service"].cat.codes
obj_df2["state_cat"] = obj_df2["state"].cat.codes

obj_df2["proto"] = obj_df2["proto_cat"]
obj_df2["service"] = obj_df2["service_cat"]
obj_df2["state"] = obj_df2["state_cat"]

obj_df2.drop('proto_cat', axis=1, inplace=True)
obj_df2.drop('service_cat', axis=1, inplace=True)
obj_df2.drop('state_cat', axis=1, inplace=True)

Y_test_all_attacks = obj_df2["attack_cat"]
obj_df2=pd.get_dummies(obj_df2, columns=["attack_cat"])


X_test = obj_df2.values[:,:-10]


for j in range(0,43):
    maximum = max(X_train[:,j])
    for i in range(0,len(X_test)):
        X_test[i,j] = round(X_test[i,j]/maximum,3)


estimators_number = list(range(11,30))

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
attack_type = 4
Y_train = obj_df.values[:,-attack_type]
Y_test = obj_df2.values[:,-attack_type]

#Run this shell only once
#Y_train=Y_train[np.newaxis]
#Y_test=Y_test[np.newaxis]
cleanup_nums = {"Worms":0, "Shellcode":1, "Reconnaissance":2, "Normal":3, "Generic":4, "Fuzzers":5, "Exploits":6, "DoS":7, "Backdoor":8, "Analysis":9}
#Y_train_all_attacks.replace(cleanup_nums,inplace=True)
#Y_test_all_attacks.replace(cleanup_nums,inplace=True)

print X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

#Measure the classification (supervised learning time)
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
clf1 = KNeighborsClassifier(); 
from sklearn.naive_bayes import GaussianNB,MultinomialNB
clf2 = sklearn.naive_bayes.GaussianNB(); 
from sklearn.ensemble import RandomForestClassifier
clf3 =  RandomForestClassifier(); 
from sklearn.neural_network import MLPClassifier
clf4 = MLPClassifier(hidden_layer_sizes=(43,1)); 
from sklearn.ensemble import GradientBoostingClassifier
clf5 = GradientBoostingClassifier(); 
from sklearn import tree
clf6 = tree.DecisionTreeClassifier(); 
from sklearn.linear_model import SGDClassifier
clf7 = SGDClassifier(); 

# empty list that will hold cv scores
cv_scores = []
model_list=[clf1,clf2,clf3,clf4,clf5,clf6,clf7]

# perform 10-fold cross validation
overall_accuracy_matrix = [None]*len(X_train)
iTERATION=0

import time as tl

print "Training: KNN Modeling"; t0 = tl.time(); clf1=clf1.fit(X_train, Y_train);   t1 = tl.time(); print 'Training Time',round(t1-t0,3),'s'
print "Training: Naive Bayes Modeling"; t0 = tl.time(); clf2=clf2.fit(X_train, Y_train);  t1 = tl.time(); print 'Training Time',round(t1-t0,3),'s'
print "Training: Random Forest Modeling"; t0 = tl.time(); clf3=clf3.fit(X_train, Y_train);  t1 = tl.time(); print 'Training Time',round(t1-t0,3),'s'
print "Training: ML Perceptron Modeling"; t0 = tl.time();  clf4=clf4.fit(X_train, Y_train);  t1 = tl.time(); print 'Training Time',round(t1-t0,3),'s'
print "Training: Gradient Boosting Modeling"; t0 = tl.time(); clf5=clf5.fit(X_train, Y_train);  t1 = tl.time(); print 'Training Time',round(t1-t0,3),'s'
print "Training: Decision Tree Modeling"; t0 = tl.time(); clf6=clf6.fit(X_train, Y_train);  t1 = tl.time(); print 'Training Time',round(t1-t0,3),'s'
print "Training: SGD Modeling"; t0 = tl.time(); clf7=clf7.fit(X_train, Y_train);  t1 = tl.time(); print 'Training Time',round(t1-t0,3),'s'
print "--------------------------- \n"


#Predicting for one dataspace with 10000 records
for clf in model_list:
    print "Validation for: ",str(clf).split("(")[0]
    t0 = tl.time();
    pred = clf.predict(X_test[:10000,:])
    t1 = tl.time(); print 'Validaton Time',round(t1-t0,5),'s'
    scores = round(accuracy_score(Y_test[:10000], pred),3)*100
    print 'Accuracy = {0}%'.format(scores)
    cv_scores.append(scores)
    
    print "-----------*----------------"
print(cv_scores)

#Measure the clustering (unsupervised learning time)
from sklearn.cluster import KMeans
print 'Kmeans'
t0 = tl.time()
kmeans = KMeans(n_clusters=6, random_state=0).fit(X_train)
t1 = tl.time(); print 'Training Time',round(t1-t0,3),'s'
t0 = tl.time()
pred = kmeans.predict(X_test[:10000,:]) #Predicting for one dataspace with 10000 records
t1 = tl.time(); print 'Validation Time',round(t1-t0,3),'s'
print "-----------*----------------"
from sklearn.cluster import Birch
print 'Birch'
t0 = tl.time()
birch = Birch(branching_factor=50, n_clusters=None, threshold=0.5, compute_labels=True).fit(X_train)
t1 = tl.time(); print 'Training Time',round(t1-t0,3),'s'
t0 = tl.time()
pred = birch.predict(X_test[:10000,:]) #Predicting for one dataspace with 10000 records
t1 = tl.time(); print 'Validation Time',round(t1-t0,3),'s'
print "-----------*----------------"
from sklearn import mixture
print 'Gaussian'
t0 = tl.time()
gmm = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(X_train)
t1 = tl.time(); print 'Training Time',round(t1-t0,3),'s'
t0 = tl.time()
gmm.predict(X_test[:10000,:]) #Predicting for one dataspace with 10000 records
t1 = tl.time(); print 'Validation Time',round(t1-t0,3),'s'

#Voting time for one dataspace with 10000 records. Let the output from MLs are available
from collections import Counter
ML1_out = np.ones((10000))
ML2_out = np.zeros((10000))
ML3_out = np.ones((10000))
ML4_out = np.zeros((10000))
ML5_out = np.zeros((10000))
ML6_out = np.zeros((10000))
ML7_out = np.ones((10000))

Final_out=[]
t0 = tl.time()
for i in range(len(ML1_out)):
    lst=[ML1_out[i], ML2_out[i], ML3_out[i], ML4_out[i], ML5_out[i], ML6_out[i], ML7_out[i]]
    most_common,num_most_common = Counter(lst).most_common(1)[0]
    Final_out.append(most_common)
    
t1 = tl.time(); print 'Voting Time',round(t1-t0,3),'s'

#Ensemble Learning 
from sklearn.ensemble import AdaBoostClassifier
print 'Adaboost'
ada_discrete = AdaBoostClassifier(algorithm="SAMME.R")
t0 = tl.time()
ada_discrete.fit(X_train, Y_train)
t1 = tl.time(); print 'Training Time',round(t1-t0,3),'s'
t0 = tl.time()
pred = ada_discrete.staged_predict(X_test)
t1 = tl.time(); print 'Validation Time',round(t1-t0,3),'s'

from xgboost import XGBClassifier
print 'Xgboost'
model = XGBClassifier()
t0 = tl.time()
model.fit(X_train, Y_train)
t1 = tl.time(); print 'Training Time',round(t1-t0,3),'s'
t0 = tl.time()
pred = model.predict(X_test)
t1 = tl.time(); print 'Validation Time',round(t1-t0,3),'s'
