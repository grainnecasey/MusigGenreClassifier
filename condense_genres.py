from difflib import SequenceMatcher
import operator

import pandas as pd
from collections import defaultdict

data_by_genre = pd.read_csv(r'dataset/data_by_genres.csv', index_col=False)

print(data_by_genre.head)
print(data_by_genre.columns)

genres_master_list = ['rock', 'pop', 'hip hop', 'classical', 'country', 'alternative', 'jazz', 'edm',
                      'punk', 'metal']
genres_synonyms = defaultdict(list)
genres_synonyms['rock'] = ['beatlesque']
genres_synonyms['pop'] = []
genres_synonyms['hip hop'] = ['rap']
genres_synonyms['classical'] = ['violin', 'baroque', 'cello', 'choir', 'chamber', 'orchestra', 'ballet',
                                'wind', 'ensemble', 'viola', 'brass', 'opera', 'soprano', 'tenor', 'fiddle',
                                'orquesta', 'piano']
genres_synonyms['country'] = ['accordeon', 'accordion', 'folk', 'western', 'singer-songwriter']
genres_synonyms['alternative'] = ['indie']
genres_synonyms['jazz'] = ['blues', 'swing', 'soul', 'funk', 'raggae', 'americana', 'afrobeat', 'dancehall']
genres_synonyms['edm'] = ['techno', 'electronic', 'acid', 'boogie', 'house', 'industrial', 'psychedelic',
                          'idm', 'disco']
genres_synonyms['punk'] = ['emo', 'hardcore']
genres_synonyms['metal'] = []
sorted_genres = defaultdict(list)

genres_dfs = defaultdict(pd.DataFrame)

# Idea:
# 1) go through and add different genres to list if it contains a master genre word or a synonym
#       a) if contains none, add to an unknown list
#       b) if contains multiple, add to both lists
#       c) find most common substrings in unknowns to try to find more meaningful synonyms to add
# 4) take average features of all sub-genres in each master genre list
# 5) write to CSV file

for index, row in data_by_genre.iterrows():
    temp = []
    for genre in genres_master_list:
        if genre in row['genres']:
            temp.append(genre)
            genres_dfs[genre] = genres_dfs[genre].append(row)
            continue
        for synonym in genres_synonyms[genre]:
            if synonym in row['genres']:
                temp.append(genre)
                genres_dfs[genre] = genres_dfs[genre].append(row)
                break
    if len(temp) == 1:
        sorted_genres[temp[0]].append(row['genres'])
    elif len(temp) > 1:
        sorted_genres['more_than_one'].append(row['genres'])
    else:
        sorted_genres['unknown'].append(row['genres'])

print("rock ", len(sorted_genres['rock']))
print("pop", len(sorted_genres['pop']))
print("hip hop", len(sorted_genres['hip hop']))
print("classical ", len(sorted_genres['classical']))
print("country", len(sorted_genres['country']))
print("alternative ", len(sorted_genres['alternative']))
print("jazz ", len(sorted_genres['jazz']))
print("edm ", len(sorted_genres['edm']))
print("punk ", len(sorted_genres['punk']))
print("metal ", len(sorted_genres['metal']))
print("more than one ", len(sorted_genres['more_than_one']))
print("unknown ", len(sorted_genres['unknown']))
print(sorted_genres['unknown'])

substring_counts = {}

for i in range(0, len(sorted_genres['unknown'])):
    for j in range(i+1,len(sorted_genres['unknown'])):
        string1 = sorted_genres['unknown'][i]
        string2 = sorted_genres['unknown'][j]
        match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
        matching_substring=string1[match.a:match.a+match.size]
        if len(matching_substring) >= 3:
            if (matching_substring not in substring_counts):
                substring_counts[matching_substring]=1
            else:
                substring_counts[matching_substring]+=1

sorted_substrings = sorted(substring_counts.items(), key=operator.itemgetter(1))
print(sorted_substrings)

#print(genres_dfs)
for key in genres_dfs.keys():
    df = genres_dfs[key]
    average_cols = df.mean(axis=0)
    print(key, "\n", average_cols.columns)