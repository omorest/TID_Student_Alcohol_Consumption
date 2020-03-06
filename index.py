#from pandas import Series, DataFrame
import pandas as pd
import numpy as np
#import os
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
#Extra
#import seaborn as sns

# Cargar ficheros de datos: Formato csv
routeFilename = "datas/student-mat.csv"
dataFramePanda = pd.read_csv(routeFilename)
#print(dataFramePanda)

# Eliminamos los valores nulos, ya que no son utiles para el arbol
dataFrameClean = dataFramePanda.dropna()

#print(dataFrameClean.dtypes)
#print(dataFrameClean.describe())

usefulVariables = ['age', 'sex']
#predictors = dataFrameClean[['Fedu', 'age']]
predictors = dataFrameClean[usefulVariables]


targets = dataFrameClean.Dalc


pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=.4)

#print(pred_train)
#print(pred_test)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape


classifier = DecisionTreeClassifier()
# Linea del error, no toma variables numericas y texto a la vez
classifier.fit(pred_train,tar_train)

"""
predictions=classifier.predict(pred_test)


sklearn.metrics.confusion_matrix(tar_test,predictions)


sklearn.metrics.accuracy_score(tar_test, predictions)


from sklearn import tree
from io import StringIO
from IPython.display import Image


out = StringIO()
tree.export_graphviz(classifier, out_file='treeMacarena.dot')
"""
