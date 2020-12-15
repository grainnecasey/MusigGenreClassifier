import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    mean_squared_error,
    accuracy_score,
    plot_confusion_matrix,
    roc_curve,
    auc
)

joined_data = pd.read_csv(r'joined_data.csv', index_col=False)
joined_data['new_genres'] = joined_data['new_genres'].str.strip()

print(joined_data.info())

print(joined_data['new_genres'].value_counts())

genres_master_list = ['rock', 'pop', 'hip hop', 'classical', 'country', 'alternative', 'jazz', 'edm', 'metal'] #'classical',

equal_dist_df = pd.DataFrame(columns=joined_data.columns)

for genre in genres_master_list:
    rows_of_genre = joined_data.loc[joined_data['new_genres'] == genre].sample(4000)
    equal_dist_df = equal_dist_df.append(rows_of_genre)

print(equal_dist_df['new_genres'].value_counts())

equal_dist_df = equal_dist_df.drop(["id", "name", "release_date", "mode", 'duration_ms'], axis = 1)

Y = equal_dist_df[equal_dist_df.columns[-1]]
X = equal_dist_df.drop(columns=[equal_dist_df.columns[-1]])

scaler = StandardScaler()
scaler.fit(X.values)

X_scaled = scaler.transform(X.values)
X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

x_train, x_test, y_train, y_test = train_test_split(X_scaled_df, Y, test_size=.25)

max_depth = [1, 5, 10, 15]




# Cross Validation
# param_grid = {'hidden_layer_sizes': [(100,), (150,)],
#               'solver': ['sgd', 'adam'],
#               'alpha': [0.0001, 0.05, 0.1],
#               'max_iter': [150, 200, 300],
#               'learning_rate': ['constant', 'adaptive']
#               }
#
# nn = MLPClassifier()
#
# nn_grid_search_cv = GridSearchCV(nn, param_grid, cv=5,
#                                  scoring="accuracy",
#                                  return_train_score=True,
#                                  verbose=True,
#                                  n_jobs=-1)
#

nn_grid_search_cv = MLPClassifier(hidden_layer_sizes=(150,), alpha=0.05, max_iter=200, solver='adam', learning_rate='adaptive')

nn_grid_search_cv.fit(x_train, y_train)

# print("CV best params:")
# print(nn_grid_search_cv.best_params_)
# print("CV best estimator")
# print(nn_grid_search_cv.best_estimator_)

# Testing

test_pred = nn_grid_search_cv.predict(x_test)
train_pred = nn_grid_search_cv.predict(x_train)
test_acc = (accuracy_score(test_pred, y_test))
test_error = 1 - test_acc
train_acc = (accuracy_score(train_pred, y_train))
train_error = (1 - train_acc)
test_prec = precision_score(y_test, test_pred, average='weighted')
test_recall = recall_score(y_test, test_pred, average='weighted')
train_prec = precision_score(y_train, train_pred, average='weighted')
train_recall = recall_score(y_train, train_pred, average='weighted')
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
print("Testing classification report:")
print(classification_report(y_test, test_pred, labels=genres_master_list))
plot_confusion_matrix(nn_grid_search_cv, x_test, y_test, labels=genres_master_list, xticks_rotation='vertical')


from sklearn.preprocessing import label_binarize

# Binarize the output
y_test_bin = label_binarize(y_test, classes = genres_master_list)
y_train_bin = label_binarize(y_train, classes = genres_master_list)

# Change whats inside this to match the model you're plotting
classifier = nn_grid_search_cv
y_score = classifier.fit(x_train, y_train_bin).predict(x_test)
n_classes = 9
print("AUC Score:", roc_auc_score(y_test_bin, y_score, average = 'micro'))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Plot ROC curve
plt.figure(figsize=(10, 10))
plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(genres_master_list[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")

plt.show()


plt.show()
