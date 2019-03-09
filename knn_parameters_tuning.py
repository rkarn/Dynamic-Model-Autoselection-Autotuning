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
from sklearn.neighbors import KNeighborsClassifier

print "KNeighborsClassifier"
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

#obj_df=pd.get_dummies(obj_df, columns=["attack_cat"])


X_train = obj_df.values[:,:-1]
Y_train = obj_df.values[:,-1]

for j in range(0,43):
    maximum = max(X_train[:,j])
    for i in range(0,len(X_train)):
        X_train[i,j] = round(X_train[i,j]/maximum,3)


# Read in the testing CSV file 
print "Reading Testing csv file."
df2 = pd.read_csv("UNSW_NB15_testing-set.csv")
df2.drop('label', axis=1, inplace=True)

obj_df=df2

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

#obj_df=pd.get_dummies(obj_df, columns=["attack_cat"])


X_test = obj_df.values[:,:-1]
Y_test = obj_df.values[:,-1]

for j in range(0,43):
    maximum = max(X_train[:,j])
    for i in range(0,len(X_test)):
        X_test[i,j] = round(X_test[i,j]/maximum,3)

# creating odd list of K for KNN
myList = list(range(1,30))

# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    clf = KNeighborsClassifier(n_neighbors=k)
    print "Creating the model for ",k," neighbors."
    clf.fit(X_train, Y_train)
    print "Testing the model for ",k," neighbors."
    pred = clf.predict(X_test)
    print "Calculating accuracy score for ",k," neighbors."
    scores = round(accuracy_score(Y_test, pred),3)*100
    cv_scores.append(scores)

print "Number of neighbors:",neighbors
print "Validation Score:",cv_scores
