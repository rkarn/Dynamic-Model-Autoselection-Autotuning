#Different algorithms are reviewed from http://scikit-learn.org/stable/modules/feature_selection.html. Some shows the improvement, while others don’t. An example for the attack type DoS with Linear SVC is shown below:
import pandas as pd
import numpy as np
import csv
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
clf = LinearSVC()
df = pd.read_csv("UNSW_NB15_training-set-formated.csv")
X=df.values[:,:-10]
y=df.values[:,-8]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
target_pred=clf.fit(X_train,y_train).predict(X_test)
print "Accuracy with full features from dataset : ",accuracy_score(y_test, target_pred, normalize = True)
print "Shape of the matrix used : ",X.shape
clf = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
print "Shape of the matrix used after feature reduction : ",X_new.shape
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.33, random_state=42)
target_pred=clf.fit(X_train,y_train).predict(X_test)
print "Accuracy with feature reduction : ",accuracy_score(y_test, target_pred, normalize = True)
