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
from sklearn.ensemble import RandomForestClassifier
clf =  RandomForestClassifier(n_estimators=140)


# Read in the CSV file 
df = pd.read_csv("UNSW_NB15_training-set.csv")
df.drop('label', axis=1, inplace=True)

obj_df=df

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


X = obj_df.values[:,:-10]



for j in range(0,43):
    maximum = max(X[:,j])
    for i in range(0,len(X)):
        X[i,j] = round(X[i,j]/maximum,3)




for i in range(1,11):

    y = obj_df.values[:,-i]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



    model = clf
    model.fit(X_train,y_train)
    target_pred = clf.predict(X_test)
    print "Accuracy score for over all model:", round(accuracy_score(y_test, target_pred, normalize = True),3)
    print "--------------*-------------*------------------------"

    def ClusterIndicesNumpy(clustNum, labels_array): #numpy 
        return np.where(labels_array == clustNum)[0]

    max_cluster_size = int(input("Enter the maximum number of clusters for unsupervised learning."))
    print "--------------*-------------*------------------------"
    overall_accuracy_matrix = [None]*100
    for cluster_size in range(2,max_cluster_size+1):
        kmeans = KMeans(n_clusters=cluster_size)
        kmeans.fit(X_train)
        #centroids = kmeans.cluster_centers_
        #labels = kmeans.labels_
        cluster_indices_train=[None]*100
        for clust in range(0,cluster_size):
            cluster_indices_train[clust] = ClusterIndicesNumpy(clust, kmeans.labels_)


        new_label = kmeans.predict(X_test)    
        cluster_indices_test=[None]*100
        for clust in range(0,cluster_size):
            cluster_indices_test[clust] = ClusterIndicesNumpy(clust, new_label)

        model_accuracy = []

        for clust in range(0,cluster_size):
            model = clf
            cluster_trainingset_X = X_train[cluster_indices_train[clust]]
            cluster_trainingset_y = y_train[cluster_indices_train[clust]]
            model=model.fit(cluster_trainingset_X,cluster_trainingset_y)
            target_pred = model.predict(X_test)
            #print "Accuracy score using cluster: {0} model on overall testing dataset:".format(clust), accuracy_score(y_test, target_pred, normalize = True)
            cluster_testingset_X = X_test[cluster_indices_test[clust]]
            cluster_testingset_y = y_test[cluster_indices_test[clust]]
            target_pred = model.predict(cluster_testingset_X)
            score=accuracy_score(cluster_testingset_y, target_pred, normalize = True)
            #print "Accuracy score using cluster: {0} model on clustered-{0} testing data:".format(clust), score
            #print "--------------*-------------*------------------------"
            model_accuracy.append(round(score,3))

        overall_accuracy_matrix[cluster_size] = model_accuracy
        print model_accuracy
        print "--------------*-------------*------------------------"
        # plot misclassification error vs n
        x = range(1,max_cluster_size+1)[0:cluster_size]
        pylab.plot(x,overall_accuracy_matrix[cluster_size],label=cluster_size)
        pylab.xlabel('Cluster Number')
        pylab.ylabel('Machine Learning Accurcy')
        pylab.legend(loc='upper right')
        plt.xticks(np.arange(min(x), max(x)+1, 1.0))
