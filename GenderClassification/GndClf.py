from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
import numpy as np

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
# Gender
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Desision Tree classifier 
clf_dt = tree.DecisionTreeClassifier()
clf_dt = clf_dt.fit(X, Y)
pred_dt = clf_dt.predict(X)
acc_dt = accuracy_score(Y, pred_dt, normalize=False)
print("The Desision tree prediction is {}".format(pred_dt) )
print("It's Accuracy is {}".format(acc_dt))

# Naive Bayes Classifier
gnb = GaussianNB()
clf_gnb = gnb.fit(X,Y,)
pred_gnb = gnb.predict(X)
acc_gnb = accuracy_score(Y, pred_gnb, normalize=False)
print("The Naive Bayes  prediction is {}".format(pred_gnb) )
print("It's Accuracy is {}".format(acc_gnb))

# Nearest Centroid Classifier
clf_nc = NearestCentroid()
clf_nc = clf_nc.fit(X, Y)
pred_nc = clf_nc.predict(X)
acc_nc = accuracy_score(Y, pred_nc, normalize=False)
print("The Nearest Centroid prediction is {}".format(pred_nc) )
print("It's Accuracy is {}".format(acc_nc))

# SVM Classifier
clf_svc = svm.LinearSVC()
clf_svc = clf_svc.fit(X, Y)
pred_svc = clf_svc.predict(X)
acc_svc = accuracy_score(Y, pred_svc, normalize=False)
print("The SVC prediction is {}".format(pred_svc))
print("It's Accuracy is {}".format(acc_svc))

# Perceptron
clf_p = Perceptron()
clf_p = clf_p.fit(X,Y)
pred_p = clf_p.predict(X)
acc_p = accuracy_score(Y,pred_p, normalize=False)
print("The Perceptron Prediction is {}".format(pred_p))
print("It's Accuracy is {}".format(acc_p))

# Best Accuracy
classifier = {0: 'Decision Tree', 1: 'Naive Bayes', 2: 'Nearest Centroid', 3: 'Perceptron', 4: 'Support Vector'}
accu = [acc_dt, acc_gnb, acc_nc, acc_p, acc_svc]
Index = accu.index(max(accu)) 
print("We can see the Best accuracy was of {}".format(classifier[Index]))

