
import pandas as pd
import numpy as np
import csv
import os

# Read in the CSV file 
df1 = pd.read_csv("UNSW_NB15_training-set.csv")
df2 = pd.read_csv("UNSW_NB15_testing-set.csv")

df = [df1, df2]
df = pd.concat(df)

print df1.shape
print df2.shape
print df.shape
print len(df1)+len(df2)
print len(df)

# Read in the CSV file 

df.drop('label', axis=1, inplace=True)
#df=pd.get_dummies(df, columns=["attack_cat"])

import category_encoders as ce

# Specify the columns to encode then fit and transform
encoder = ce.backward_difference.BackwardDifferenceEncoder(cols=["proto","service","state"])
encoder.fit(df, verbose=1)

df=encoder.transform(df).iloc[:,:]
array = df.values
print("Creating the UNSW_NB15_training-set-formated.csv file.")
new_dataframe = pd.DataFrame(data=array[1:82332,1:], index=array[1:82332,0], columns=array[0,1:])
new_dataframe.to_csv("UNSW_NB15_training-set-formated.csv", sep=',', encoding='utf-8')
print("Done.")
print("Creating the UNSW_NB15_testing-set-formated.csv file.")
new_dataframe = pd.DataFrame(data=array[82333:,1:], index=array[82333:,0], columns=array[0,1:])
new_dataframe.to_csv("UNSW_NB15_testing-set-formated.csv", sep=',', encoding='utf-8')
print("Done.")

from matplotlib.pyplot import * 
import matplotlib.pyplot as plt
import pylab
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.cluster import KMeans
pylab.rcParams['figure.figsize'] = (16.0, 5.0)

df = pd.read_csv("UNSW_NB15_training-set-formated.csv")
import sklearn
from sklearn.naive_bayes import GaussianNB,MultinomialNB
clf = sklearn.naive_bayes.GaussianNB() 


X=df.values[:,:-1]
y=df.values[:,-1]


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
    
pylab.show()
