from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load dataset
df = pd.read_csv("movies.csv")

# Column name in your dataset
TITLE_COL = "original_title"

# Create merged content column
df["content"] = df.fillna("").astype(str).apply(" ".join, axis=1)

# Create recommendation model
cv = CountVectorizer(stop_words="english")
vectors = cv.fit_transform(df["content"])
similarity = cosine_similarity(vectors)

# Function to simplify filenames (remove spaces/special chars)
def clean_name(title):
    return "".join(c.lower() if c.isalnum() else "_" for c in title)

# Recommendation
def recommend(movie):
    titles = df[TITLE_COL].str.lower()
    movie = movie.lower()
    if movie not in titles.values:
        return None
    
    index = titles[titles == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recs = df.iloc[[i[0] for i in movie_list]][TITLE_COL].values
    return recs

app = Flask(__name__)

# Home Page
@app.route("/", methods=["GET", "POST"])
def home():
    all_movies = sorted(df[TITLE_COL].dropna().unique())
    if request.method == "POST":
        movie = request.form.get("movie")
        results = recommend(movie)
        return render_template("recommend.html", movie=movie, results=results)
    return render_template("index.html", movies=all_movies)

# Dynamic poster fetch
@app.template_filter("poster")
def get_poster(movie_title):
    filename = clean_name(movie_title) + ".jpg"
    path = os.path.join("static", "posters", filename)
    if os.path.exists(path):
        return f"/static/posters/{filename}"
    else:
        return "/static/posters/default.jpg"  # Put a default image here

if __name__ == "__main__":
    app.run(debug=True)
