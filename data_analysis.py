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

not_cat_data = joined_data.select_dtypes(exclude=['object']).copy()

# for col in not_cat_data.columns:
#     if col == 'new_genres':
#         continue
#     joined_data.boxplot(column=col, by='new_genres')

print(joined_data['new_genres'].value_counts())

# plt.show()