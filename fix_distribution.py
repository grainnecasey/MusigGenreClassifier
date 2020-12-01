from difflib import SequenceMatcher
import operator

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

joined_data = pd.read_csv(r'joined_data.csv', index_col=False)
joined_data['new_genres'] = joined_data['new_genres'].str.strip()

print(joined_data.info())

print(joined_data['new_genres'].value_counts())

genres_master_list = ['rock', 'pop', 'hip hop', 'classical', 'country', 'alternative', 'jazz', 'edm', 'metal']

equal_dist_df = pd.DataFrame(columns=joined_data.columns)

for genre in genres_master_list:
    rows_of_genre = joined_data.loc[joined_data['new_genres'] == genre].sample(1000)
    equal_dist_df = equal_dist_df.append(rows_of_genre)

print(equal_dist_df['new_genres'].value_counts())





