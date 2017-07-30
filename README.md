Gender Classification with Scikit Learn
=======================================

In This Program We are Classifying the Gender Of people By Knowing the Body metrics only..
We are going to make The Training Dataset with the help of 2 Lists
List X contains the Body metrics [height, weight, foot size]
List Y contains the Gender 

> Currently Due to lack of test data we are using  the same train dataset.


*Dependencies*
--------------

	 1. Scikit Learn 
	 2. Numpy(meh i didn't need it :P)

Classification Models Used
-------

***1. Decision Tree Classifier***

Decision trees are a widely used models for classification and regression tasks. Essentially, they learn a hierarchy of “if-else” questions, leading to a decision. 

***2. Gaussian Naive Bayes Classifier***

Naive Bayes classifiers are a family of classifiers that are quite similar to the linear models. They tend to be faster in training.
Naive Bayes models often provide generalization performance that is slightly worse than linear classifiers like LogisticRegression and Line arSVC. 
GaussianNB implements the Gaussian Naive Bayes algorithm for classification. The likelihood of the features is assumed to be Gaussian.

***3. Nearest Centroid***

The NearestCentroid classifier is a simple algorithm that represents each class by the centroid of its members. In effect, this makes it similar to the label updating phase of the sklearn.KMeans algorithm. It also has no parameters to choose, making it a good baseline classifier.

***4. Perceptron***

Perceptron is a type of linear classifier.

***5. Support Vector Classifier***

SVC and NuSVC are methods, that accepts slightly different sets of parameters and have different mathematical formulations. LinearSVC is another implementation of Support Vector Classification for the case of a linear kernel. 
Linear Support Vector Classification, is similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples. All three are Based On *Support Vector Machines Model*

Accuracy Metrics
-------
To measure and Compare the Accuracy of the Prediction we are using the accuracy_score fuction. We shall measure the accuracy via the count of nomber of right predictions.
The accuracy_score function can computes the accuracy, by the count of correct predictions, if implemented as
accuracy_score(X, Y, normalize=False)


![](http://scikit-learn.org/stable/_images/math/cd4bea15b385d15cceb8e24f68976da7d8510290.png)
[See this](http://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score)

Output
-------
	The Desision tree prediction is ['male' 'male' 'female' 'female' 'male' 'male' 'female' 'female' 'female'
	 'male' 'male']

	It's Accuracy is 11

	The Naive Bayes  prediction is ['male' 'male' 'female' 'female' 'female' 'male' 'female' 'male' 'female'
	 'male' 'male']

	It's Accuracy is 9

	The Nearest Centroid prediction is ['male' 'male' 'female' 'female' 'female' 'male' 'female' 'male' 'female'
	 'male' 'male']

	It's Accuracy is 9

	The SVC prediction is ['male' 'male' 'female' 'female' 'female' 'male' 'female' 'female' 'female'
	 'male' 'male']

	It's Accuracy is 10

	The Perceptron Prediction is ['male' 'male' 'male' 'male' 'male' 'male' 'male' 'male' 'male' 'male'
	 'male']

	It's Accuracy is 6

	We can see the Best accuracy was of Decision Tree
