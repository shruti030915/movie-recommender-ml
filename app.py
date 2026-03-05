import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommendation System")

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

data = pd.merge(ratings, movies, on="movieId")

user_movie_matrix = data.pivot_table(index="userId", columns="title", values="rating")
user_movie_matrix_filled = user_movie_matrix.fillna(0)

movie_similarity = cosine_similarity(user_movie_matrix_filled.T)

movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=user_movie_matrix.columns,
    columns=user_movie_matrix.columns
)

def recommend_movies(movie_name):

    similar_movies = movie_similarity_df[movie_name]
    similar_movies = similar_movies.sort_values(ascending=False)

    return similar_movies.iloc[1:11]


movie_list = user_movie_matrix.columns.tolist()

selected_movie = st.selectbox("Select a movie:", movie_list)

if st.button("Recommend"):

    recommendations = recommend_movies(selected_movie)

    st.write("### Recommended Movies:")

    for movie in recommendations.index:
        st.write(movie)
