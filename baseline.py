# based on
# https://raw.githubusercontent.com/cahya-wirawan/ML-Collection/master/TextClassification.py


#from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.externals import joblib

import numpy as np
import data_helpers

random_state = 42

""" Load newsdata """
x_texts, y = data_helpers.load_newsdata_and_labels()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    x_texts, y, test_size=0.2, random_state=42)



""" Naive Bayes classifier """
y_test = np.argmax(y_test, axis=1) # from 1-hot of 5 to 1 scalar of 0-4
y_train = np.argmax(y_train, axis=1)

bayes_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB())
                      ])
bayes_clf.fit(X_train, y_train)
joblib.dump(bayes_clf, "baseline_bayes_newsdata.pkl", compress=9)
""" Predict the test dataset using Naive Bayes"""
predicted = bayes_clf.predict(X_test)
print('Naive Bayes correct prediction: {:4.2f}'.format(np.mean(predicted == y_test)))
print(metrics.classification_report(y_test, predicted, target_names=["pos", "slight pos", "neut", "slight neg", "neg"]))

""" Support Vector Machine (SVM) classifier"""
svm_clf = Pipeline([('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=   5, random_state=42)),
])
svm_clf.fit(X_train, y_train)
joblib.dump(svm_clf, "baseline_svm_newsdata.pkl", compress=9)
""" Predict the test dataset using Naive Bayes"""
predicted = svm_clf.predict(X_test)
print('SVM correct prediction: {:4.2f}'.format(np.mean(predicted == y_test)))
print(metrics.classification_report(y_test, predicted, target_names=["pos", "slight pos", "neut", "slight neg", "neg"]))

print(metrics.confusion_matrix(y_test, predicted))
