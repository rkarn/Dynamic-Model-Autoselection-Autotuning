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
from sklearn import cluster
from sklearn import mixture
pylab.rcParams['figure.figsize'] = (16.0, 5.0)


import sklearn
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier()

print "MLPClassifier"
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


for i in [4,6,7,8]:  #For NO-ATTACK, Fuzzers, Exploits, DoS

    Y_train = obj_df.values[:,-i]
    Y_test = obj_df2.values[:,-i]
    
    model = clf
    model.fit(X_train,Y_train)
    target_pred = clf.predict(X_test)
    print "--------------*----Attack Type: {0}-----*----------------".format(i)
    print "Accuracy score without clustering the data:", round(accuracy_score(Y_test, target_pred, normalize = True),3)
    print "--------------*-------------*------------------------"

    def ClusterIndicesNumpy(clustNum, labels_array): #numpy 
        return np.where(labels_array == clustNum)[0]

    max_cluster_size = 6 #int(input("Enter the maximum number of clusters for unsupervised learning."))
    overall_accuracy_matrix = [None]*100
    for cluster_size in range(2,max_cluster_size+1):
        #kmeans = KMeans(n_clusters=cluster_size)
        #print "Performing Clustering for attack: ",i
        #kmeans = cluster.Birch(n_clusters=cluster_size)
        kmeans = mixture.GaussianMixture(n_components=cluster_size, covariance_type='full')
        kmeans.fit(X_train)
        #centroids = kmeans.cluster_centers_
        #labels = kmeans.labels_
        cluster_indices_train=[None]*100
        for clust in range(0,cluster_size):
            #cluster_indices_train[clust] = ClusterIndicesNumpy(clust, kmeans.labels_)
            cluster_indices_train[clust] = ClusterIndicesNumpy(clust, kmeans.predict(X_train))

        #print "Predicting Clusters for attack: ",i
        new_label = kmeans.predict(X_test)
        cluster_indices_test=[None]*100
        for clust in range(0,cluster_size):
            cluster_indices_test[clust] = ClusterIndicesNumpy(clust, new_label)

        model_accuracy = []

        for clust in range(0,cluster_size):
            model = clf
            cluster_trainingset_X = X_train[cluster_indices_train[clust]]
            cluster_trainingset_y = Y_train[cluster_indices_train[clust]]
            model=model.fit(cluster_trainingset_X,cluster_trainingset_y)
            target_pred = model.predict(X_test)
            #print "Accuracy score using cluster: {0} model on overall testing dataset:".format(clust), accuracy_score(y_test, target_pred, normalize = True)
            cluster_testingset_X = X_test[cluster_indices_test[clust]]
            cluster_testingset_y = Y_test[cluster_indices_test[clust]]
            if (len(cluster_testingset_X) > 1):
                target_pred = model.predict(cluster_testingset_X)
                score=accuracy_score(cluster_testingset_y, target_pred, normalize = True)
            else:
                score = 1
            #print "Accuracy score using cluster: {0} model on clustered-{0} testing data:".format(clust), score
            #print "--------------*-------------*------------------------"
            model_accuracy.append(round(score,3))

        overall_accuracy_matrix[cluster_size] = model_accuracy
        print "Accuracy Score with {0} clusters:".format(cluster_size),model_accuracy
        print "--------------*-------------*------------------------"

