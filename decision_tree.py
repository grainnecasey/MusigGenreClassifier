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
)
from sklearn.metrics import classification_report

joined_data = pd.read_csv(r'joined_data.csv', index_col=False)
joined_data['new_genres'] = joined_data['new_genres'].str.strip()

print(joined_data.info())

print(joined_data['new_genres'].value_counts())

genres_master_list = ['rock', 'pop', 'hip hop', 'country', 'alternative', 'jazz', 'edm', 'metal'] #'classical',

equal_dist_df = pd.DataFrame(columns=joined_data.columns)

for genre in genres_master_list:
    rows_of_genre = joined_data.loc[joined_data['new_genres'] == genre].sample(4000)
    equal_dist_df = equal_dist_df.append(rows_of_genre)

print(equal_dist_df['new_genres'].value_counts())

equal_dist_df = equal_dist_df.drop(["id", "name", "release_date", "year", "mode", 'duration_ms', 'liveness'], axis = 1)

Y = equal_dist_df[equal_dist_df.columns[-1]]
X = equal_dist_df.drop(columns=[equal_dist_df.columns[-1]])

scaler = StandardScaler()
scaler.fit(X.values)

X_scaled = scaler.transform(X.values)
X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

x_train, x_test, y_train, y_test = train_test_split(X_scaled_df, Y, test_size=.25)

max_depth = [1, 5, 10, 15, 20, 30]

for depth in max_depth:
    print("======================")
    print("Depth = ", depth)

    decision_tree = DecisionTreeClassifier(max_depth=depth)
    decision_tree.fit(x_train, y_train)

    print("Training:")
    y_predict_train = decision_tree.predict(x_train)
    train_df = pd.DataFrame(data=y_predict_train)
    # print("predictions:")
    #     # print(train_df.value_counts())
    #     # print("actual:")
    #     # print(y_train.value_counts())
    print("Training classification report:")
    print(classification_report(y_train, y_predict_train, labels=genres_master_list))
    df = pd.DataFrame({"Actual": y_train, "Predicted": y_predict_train})
    print(df.head())
    train_error = 1 - accuracy_score(y_train, y_predict_train)
    print("train error", train_error)

    print("Testing:")
    y_predict_test = decision_tree.predict(x_test)
    test_df = pd.DataFrame(data=y_predict_test)
    # print("predictions:")
    # print(test_df.value_counts())
    # print("actual:")
    # print(y_test.value_counts())
    print(classification_report(y_test, y_predict_test, labels=genres_master_list))
    df = pd.DataFrame({"Actual": y_test, "Predicted": y_predict_test})
    print(df.head())
    test_error = 1 - accuracy_score(y_test, y_predict_test)
    print("test error", test_error)
    print("depth: ", decision_tree.get_depth())