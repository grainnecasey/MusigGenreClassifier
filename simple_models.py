#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import random
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import plot_roc_curve, roc_auc_score

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    mean_squared_error,
    accuracy_score
)



# In[2]:


data = pd.read_csv("joined_data.csv")

genres_master_list = ['rock', 'pop', 'hip hop', 'classical', 'country', 'alternative', 'jazz', 'edm', 'metal'] #

equal_dist_df = pd.DataFrame(columns=data.columns)

for genre in genres_master_list:
    rows_of_genre = data.loc[data['new_genres'] == genre].sample(4000)
    equal_dist_df = equal_dist_df.append(rows_of_genre)

print(equal_dist_df['new_genres'].value_counts())


data = equal_dist_df

data = data.drop(["id", "name", "release_date", "year", "mode", 'duration_ms', 'liveness'], axis = 1)
X = data.iloc[:,:-1]
y = data["new_genres"]


# In[3]:


print(data["new_genres"].unique())


# In[4]:


sss = StratifiedShuffleSplit(n_splits=1, test_size = 0.25, random_state = 42)

for train_index, test_index in sss.split(X, y):
    X_train = np.array(X)[train_index]
    X_test = np.array(X)[test_index]
    y_train = np.array(y)[train_index]
    y_test = np.array(y)[test_index]


X_train = preprocessing.scale(X_train)


# In[ ]:


clf = LogisticRegression(random_state=0).fit(X_train, y_train)
clf.score(X_train, y_train)

test_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)
test_acc = (accuracy_score(test_pred, y_test))
test_error = 1 - test_acc
train_acc = (accuracy_score(train_pred, y_train))
train_error = (1 - train_acc)
test_prec = precision_score(y_test, test_pred, average='weighted')
test_recall = recall_score(y_test, test_pred, average='weighted')
train_prec = precision_score(y_train, train_pred, average='weighted')
train_recall = recall_score(y_train, train_pred, average='weighted')

print("Training snapshot")
df = pd.DataFrame({"Actual": y_train, "Predicted": train_pred})
print(df.head())
print("Training classification report:")
print(classification_report(y_train, train_pred, labels=genres_master_list))

print("Testing snapshot")
df = pd.DataFrame({"Actual": y_test, "Predicted": test_pred})
print(df.head())
print("Testing classification report:")
print(classification_report(y_test, test_pred, labels=genres_master_list))

print("Train Accuracy:", train_acc)
print("Train Error:", train_error)
print("Train Recall:", train_recall)
print("Train Precision:", train_prec)
print("-----------------------------------------")
print("Test Accuracy:", test_acc)
print("Test Error:", test_error)
print("Test Recall:", test_recall)
print("Test Precision:", test_prec)
print("-----------------------------------------")


#In[ ]:

print("==================== Ada Boost ====================")
n_classifiers = [10, 50, 100]

for n in n_classifiers:
    ada = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth = 1),
        n_estimators = n, learning_rate = 1, random_state = 1)
    ada.fit(X_train, y_train)
    test_pred = ada.predict(X_test)
    train_pred = ada.predict(X_train)
    test_acc = (accuracy_score(test_pred, y_test))
    test_error = 1 - test_acc
    train_acc = (accuracy_score(train_pred, y_train))
    train_error = (1 - train_acc)
    test_prec = precision_score(y_test, test_pred, average='weighted')
    test_recall = recall_score(y_test, test_pred, average='weighted')
    train_prec = precision_score(y_train, train_pred, average='weighted')
    train_recall = recall_score(y_train, train_pred, average='weighted')
    print("Training snapshot")
    df = pd.DataFrame({"Actual": y_train, "Predicted": train_pred})
    print(df.head())
    print("Training classification report:")
    print(classification_report(y_train, train_pred, labels=genres_master_list))

    print("Testing snapshot")
    df = pd.DataFrame({"Actual": y_test, "Predicted": test_pred})
    print(df.head())
    print("Testing classification report:")
    print(classification_report(y_test, test_pred, labels=genres_master_list))

    print(n, "Classifiers:")
    print("Test Accuracy:", test_acc)
    print("Test Error:", test_error)
    print("Test Recall:", test_recall)
    print("Test Precision:", test_prec)
    print("-----------------------------------------")
    print("Train Accuracy:", train_acc)
    print("Train Error:", train_error)
    print("Train Recall:", train_recall)
    print("Train Precision:", train_prec)
    print("-----------------------------------------")
    print("\n")
#
#
# # In[ ]:
#
#
# n_trees = [50, 100, 150, 300, 500]
#
# for n in n_trees:
#     rf = RandomForestClassifier(n_estimators = n).fit(X_train, y_train)
#     test_pred = rf.predict(X_test)
#     train_pred = rf.predict(X_train)
#     test_acc = (accuracy_score(test_pred, y_test))
#     test_error = 1 - test_acc
#     train_acc = (accuracy_score(train_pred, y_train))
#     train_error = (1 - train_acc)
#     #test_prec = precision_score(y_test, test_pred)
#     #test_recall = recall_score(y_test, test_pred)
#     #train_prec = precision_score(y_train, train_pred)
#     #train_recall = recall_score(y_train, train_pred)
#     print(n, "Estimators:")
#     print("Test Accuracy:", test_acc)
#     print("Test Error:", test_error)
#     #print("Test Recall:", test_recall)
#     #print("Test Precision:", test_prec)
#     print("-----------------------------------------")
#     print("Train Accuracy:", train_acc)
#     print("Train Error:", train_error)
#     #print("Train Recall:", train_recall)
#     #print("Train Precision:", train_prec)
#     print("-----------------------------------------")
#     print("\n")
#
#
# # In[ ]:
#
#
# from sklearn.neural_network import MLPClassifier
# nn = MLPClassifier().fit(X_train, y_train)
#
# test_pred = nn.predict(X_test)
# train_pred = nn.predict(X_train)
# test_acc = (accuracy_score(test_pred, y_test))
# test_error = 1 - test_acc
# train_acc = (accuracy_score(train_pred, y_train))
# train_error = (1 - train_acc)
# test_prec = precision_score(y_test, test_pred, average='weighted')
# test_recall = recall_score(y_test, test_pred, average='weighted')
# train_prec = precision_score(y_train, train_pred, average='weighted')
# train_recall = recall_score(y_train, train_pred, average='weighted')
# print("Train Accuracy:", train_acc)
# print("Train Error:", train_error)
# print("Train Recall:", train_recall)
# print("Train Precision:", train_prec)
# print("-----------------------------------------")
# print("Test Accuracy:", test_acc)
# print("Test Error:", test_error)
# print("Test Recall:", test_recall)
# print("Test Precision:", test_prec)
# print("-----------------------------------------")
#
#
# # In[ ]:
#
#
# from sklearn.neural_network import MLPClassifier
# nn = MLPClassifier().fit(X_train, y_train)
#
# test_pred = nn.predict(X_test)
# train_pred = nn.predict(X_train)
# test_acc = (accuracy_score(test_pred, y_test))
# test_error = 1 - test_acc
# train_acc = (accuracy_score(train_pred, y_train))
# train_error = (1 - train_acc)
# test_prec = precision_score(y_test, test_pred, average='weighted')
# test_recall = recall_score(y_test, test_pred, average='weighted')
# train_prec = precision_score(y_train, train_pred, average='weighted')
# train_recall = recall_score(y_train, train_pred, average='weighted')
# print("Train Accuracy:", train_acc)
# print("Train Error:", train_error)
# print("Train Recall:", train_recall)
# print("Train Precision:", train_prec)
# print("-----------------------------------------")
# print("Test Accuracy:", test_acc)
# print("Test Error:", test_error)
# print("Test Recall:", test_recall)
# print("Test Precision:", test_prec)
# print("-----------------------------------------")
#
