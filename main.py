import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import time
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve


from datapipeline import *
from random_forest import *
from decision_tree import *

X_train, X_test, y_train, y_test = transform('../data/raw/data.csv')

# Testing tree

# tree = Decision_Tree(3)

# Testing Forest

rf = Random_Forest(3, 5, 500, 5)
rf.fit(X_train, y_train)

print("Training Prediction")
predictions_train = rf.predict(X_train)
print((predictions_train == y_train).sum() / len(predictions_train))

print("Testing Prediction")
predictions_test = rf.predict(X_test)
print((predictions_test == y_test).sum() / len(predictions_test))

cm = confusion_matrix(y_test, predictions_test)

plt.figure(figsize = (10,7))

plt.ylabel('Predicted label')
plt.xlabel('Predicted label')

sb.heatmap(pd.DataFrame(cm), annot=True)
plt.show()

# Manually calculate Precision and Recall using the numbers from the confusion matrix directly
tn, fp, fn, tp = cm.ravel()

my_precision_score = tp / (tp + fp)
my_recall_score = tp / (tp + fn)
my_f1_score = (2 * my_precision_score * my_recall_score) / (my_precision_score + my_recall_score)
print("Precision: " + str(my_precision_score))
print("Recall: " + str(my_recall_score))
print("F1: " + str(my_f1_score))
