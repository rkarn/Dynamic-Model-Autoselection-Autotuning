import pandas as pd
import numpy as np
import csv
import os
from matplotlib.pyplot import * 
import matplotlib.pyplot as plt
import pylab
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.cluster import KMeans
pylab.rcParams['figure.figsize'] = (16.0, 5.0)


import sklearn
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()

print "SGDClassifier"
print "-----------------------------------------------------"
# Read in the training CSV file
print "Reading Training csv file." 
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

obj_df=pd.get_dummies(obj_df, columns=["attack_cat"])


X_train = obj_df.values[:,:-10]


for j in range(0,43):
    maximum = max(X_train[:,j])
    for i in range(0,len(X_train)):
        X_train[i,j] = round(X_train[i,j]/maximum,3)


# Read in the testing CSV file 
print "Reading Testing csv file."
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

obj_df2=pd.get_dummies(obj_df2, columns=["attack_cat"])


X_test = obj_df2.values[:,:-10]


for j in range(0,43):
    maximum = max(X_test[:,j])
    for i in range(0,len(X_test)):
        X_test[i,j] = round(X_test[i,j]/maximum,3)


estimators_number = list(range(10,30))

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for attack_type in range(1,11):
    Y_train = obj_df.values[:,-attack_type]
    Y_test = obj_df2.values[:,-attack_type]
    print "Creating the model for attack: ",attack_type
    clf.fit(X_train, Y_train)
    print "Testing the model for attack: ",attack_type
    pred = clf.predict(X_test)
    print "Calculating accuracy score for attack: ",attack_type
    scores = round(accuracy_score(Y_test, pred),3)*100
    cv_scores.append(scores)

print "Validation Score:",cv_scores
