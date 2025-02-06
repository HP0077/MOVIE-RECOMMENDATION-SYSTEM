from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from thefuzz import process  # Use thefuzz (newer version of fuzzywuzzy)
except ImportError:
    from fuzzywuzzy import process  # Fallback to fuzzywuzzy if thefuzz isn't available
import os

app = Flask(__name__, template_folder="templates")
CORS(app)

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

# Function to get recommendations using fuzzy matching & TF-IDF
def recommend_movie(movie_name):
    movie_name = movie_name.lower().strip()

    print("🔍 Searching for:", movie_name)

    # Use Fuzzy Matching to find the best matching movie title
    match = process.extractOne(movie_name, movies_data['title_lower'])

    if match:  # Ensure we got a valid match
        best_match = match[0]  # Movie title
        score = match[1]       # Matching score
    else:
        print("❌ No strong match found!")
        return []

    if score < 60:  # If the match is weak, return an empty list
        print("❌ No strong match found!")
        return []

    print(f"✅ Best match found: {best_match} (Score: {score})")

    # Find the index of the best-matched movie
    movie_index = movies_data[movies_data['title_lower'] == best_match].index[0]

    # Find top 25 most similar movies based on descriptions
    scores = sorted(list(enumerate(similarity[movie_index])), key=lambda x: x[1], reverse=True)[1:25]

    recommended_movies = [movies_data.iloc[i[0]]['title'] for i in scores]

    print("🎥 Recommendations:", recommended_movies)
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
