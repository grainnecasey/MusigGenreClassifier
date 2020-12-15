import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    multilabel_confusion_matrix,
    plot_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.metrics import classification_report

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

equal_dist_df = equal_dist_df.drop(["id", "name", "release_date", "mode", 'duration_ms', 'liveness', 'key'], axis = 1)

Y = equal_dist_df[equal_dist_df.columns[-1]]
X = equal_dist_df.drop(columns=[equal_dist_df.columns[-1]])

scaler = StandardScaler()
scaler.fit(X.values)

X_scaled = scaler.transform(X.values)
X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

x_train, x_test, y_train, y_test = train_test_split(X_scaled_df, Y, test_size=.25)

max_depth = [1, 5, 10, 15]
#max_depth = [10]

# for depth in max_depth:
#     print("======================")
#     print("Depth = ", depth)
#
#     decision_tree = DecisionTreeClassifier(max_depth=depth)
#     decision_tree.fit(x_train, y_train)
#
#     print("Training:")
#     y_predict_train = decision_tree.predict(x_train)
#     train_df = pd.DataFrame(data=y_predict_train)
#     # print("predictions:")
#     #     # print(train_df.value_counts())
#     #     # print("actual:")
#     #     # print(y_train.value_counts())
#     print("Training classification report:")
#     print(classification_report(y_train, y_predict_train, labels=genres_master_list))
#     df = pd.DataFrame({"Actual": y_train, "Predicted": y_predict_train})
#     print(df.head())
#     train_error = 1 - accuracy_score(y_train, y_predict_train)
#     print("train accuracy", accuracy_score(y_train, y_predict_train))
#     print("train error", train_error)
#     train_prec = precision_score(y_train, y_predict_train, average='weighted')
#     train_recall = recall_score(y_train, y_predict_train, average='weighted')
#     print("train precision:", train_prec)
#     print("train recall:", train_recall)
#
#     print("Testing:")
#     y_predict_test = decision_tree.predict(x_test)
#     test_df = pd.DataFrame(data=y_predict_test)
#     print("predictions:")
#     print(test_df.value_counts())
#     print("actual:")
#     print(y_test.value_counts())
#     print("Testing classification report:")
#     print(classification_report(y_test, y_predict_test, labels=genres_master_list))
#     plot_confusion_matrix(decision_tree, x_test, y_test, labels=genres_master_list, xticks_rotation='vertical')
#     # print("Testing confusion matrix:")
#     # test_confusion_matrix = confusion_matrix(y_test, y_predict_test, labels=genres_master_list)
#     # print(confusion_matrix(y_test, y_predict_test))
#     # print("multilabel confusion matrix")
#     # print(genres_master_list)
#     # print(multilabel_confusion_matrix(y_test, y_predict_test, labels=genres_master_list))
#     df = pd.DataFrame({"Actual": y_test, "Predicted": y_predict_test})
#     print(df.head())
#     test_error = 1 - accuracy_score(y_test, y_predict_test)
#     test_prec = precision_score(y_test, y_predict_test, average='weighted')
#     test_recall = recall_score(y_test, y_predict_test, average='weighted')
#     print("test accuracy", accuracy_score(y_test, y_predict_test))
#     print("test error", test_error)
#     print("test precision:", test_prec)
#     print("test recall:", test_recall)
#     print("depth: ", decision_tree.get_depth())


from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import label_binarize

# Binarize the output
y_test_bin = label_binarize(y_test, classes = genres_master_list)
y_train_bin = label_binarize(y_train, classes = genres_master_list)

# Change whats inside this to match the model you're plotting
classifier = DecisionTreeClassifier(max_depth=10)
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