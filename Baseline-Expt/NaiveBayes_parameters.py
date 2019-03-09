#2.	Naïve Bayes Algorithm: No any parameters can be set externally. The sklearn takes only two parameters as X and Y. Perform the accuracy testing as well as 10-fold cross validation, the accuracy is approximately 41%.

import pandas as pd
import numpy as np
import csv
import os
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB,MultinomialNB

# Read in the CSV file 
df1 = pd.read_csv("UNSW_NB15_training-set-formated.csv")

X=df1.values[:,:-1]
y=df1.values[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
cfr = sklearn.naive_bayes.GaussianNB() 

cfr.fit(X_train,y_train)

target_pred = cfr.predict(X_test)

print accuracy_score(y_test, target_pred, normalize = True)

#10 fold cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(cfr, X_train, y_train, cv=10, scoring='accuracy')
print scores.mean()
