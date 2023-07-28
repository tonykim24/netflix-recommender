from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

movies = pd.read_csv('tmdb_5000_movies.csv')
movies.head(10)

# see the data statistics(ex. mean, std, etc)
movies.describe()

movies.columns

movies = movies[['id', 'original_title', 'overview', 'genres']]

# create a new column called "tags"
movies['tags'] = movies['overview']+movies['genres']
newData = movies.drop(columns=['overview', 'genres'])
print(newData)

cv = CountVectorizer(max_features=10000, stop_words='english')
vector = cv.fit_transform(newData['tags'].values.astype('U')).toarray()

similarity = cosine_similarity(vector)

# index = newData[newData['original_title'] == "Iron Man"].index[0]
# distance = sorted(
#     list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
# for i in distance[0:5]:
#     print(newData.iloc[i[0]].original_title)


def recommend(movieName):
    index = newData[newData['original_title'] == movieName].index[0]
    distance = sorted(
        list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
    for i in distance[1:6]:
        print(newData.iloc[i[0]].original_title)


recommend("Iron Man")

print("test run")
