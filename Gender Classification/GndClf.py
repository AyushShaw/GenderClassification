from sklearn import tree
from sklearn import svm


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
# Gender
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#Desision Tree classifier 
clf_t = tree.DecisionTreeClassifier()

clf_t = clf_t.fit(X, Y)

pred_t = clf_t.predict([[143, 50, 33]])

print("The Desision tree prediction is {}".format(pred_t) )

# SVM Classifier
clf_svc = svm.LinearSVC()

clf_svc = clf_svc.fit(X, Y)

pred_svc = clf_svc.predict([[143, 50, 33]])

print("The SVC prediction is {}".format(pred_svc) )