from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__, template_folder="templates")
CORS(app)  # ✅ Fixes frontend-backend communication issues

# Load the dataset
movies_file_path = os.path.join(os.path.dirname(__file__), 'movies.csv')

if not os.path.exists(movies_file_path):
    raise FileNotFoundError("movies.csv file not found! Make sure it's in the correct folder.")

movies_data = pd.read_csv(movies_file_path)
movies_data.fillna('unknown', inplace=True)

movies_data['combined_features'] = (
    movies_data['title'].fillna('') + " " +
    movies_data['genres'].fillna('') + " " +
    movies_data['overview'].fillna('')
)

# Convert to lowercase for better matching
movies_data['title_lower'] = movies_data['title'].str.lower()

# Create similarity matrix
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
feature_vectors = vectorizer.fit_transform(movies_data['combined_features'])
similarity = cosine_similarity(feature_vectors)

# Function to get recommendations
def recommend_movie(movie_name):
    movie_name = movie_name.lower().strip()

    print("🔍 Searching for:", movie_name)  # Debugging

    # Convert movie titles to lowercase for comparison
    movies_data['title_lower'] = movies_data['title'].str.lower()

    matching_movies = movies_data[movies_data['title_lower'].str.contains(movie_name, regex=False, na=False)]

    if matching_movies.empty:
        print("❌ Movie not found in dataset!")
        return []

    movie_index = matching_movies.index[0]  # Get the first matching movie
    scores = sorted(list(enumerate(similarity[movie_index])), key=lambda x: x[1], reverse=True)[1:6]

    recommended_movies = [movies_data.iloc[i[0]]['title'] for i in scores]

    print("🎥 Recommendations:", recommended_movies)  # Debugging
    return recommended_movies
# Homepage route
@app.route('/')
def home():
    return render_template('index.html')

# API route to get recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        movie_name = data.get("movie", "").strip()

        if not movie_name:
            return jsonify({"error": "No movie name provided!"}), 400

        recommendations = recommend_movie(movie_name)
        return jsonify({"recommendations": recommendations})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
