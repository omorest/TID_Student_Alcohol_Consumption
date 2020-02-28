from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

AH_data = pd.read_csv("datas/student-mat.csv")


data_clean = AH_data.dropna()

data_clean.dtypes

data_clean.describe()


predictors = data_clean[['Fedu', 'age']]

targets = data_clean.Dalc


pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=.4)


pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape


classifier = DecisionTreeClassifier()
classifier.fit(pred_train,tar_train)


predictions=classifier.predict(pred_test)


sklearn.metrics.confusion_matrix(tar_test,predictions)


sklearn.metrics.accuracy_score(tar_test, predictions)


from sklearn import tree
from io import StringIO
from IPython.display import Image


out = StringIO()
tree.export_graphviz(classifier, out_file='treeMacarena.dot')
