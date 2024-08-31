
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Load the datasets
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv(r'C:\Users\USER\Documents\MyPythonProjects\ml-1m\ratings.dat', names=column_names, sep='::', 
                     header=None, encoding='ISO-8859-1', engine='python')

movies = pd.read_csv(r'C:\Users\USER\Documents\MyPythonProjects\ml-1m\movies.dat', sep='::', 
                     header=None, encoding='ISO-8859-1', engine='python')
movies.columns = ['MovieID', 'title', 'Genres']

users = pd.read_csv(r'C:\Users\USER\Documents\MyPythonProjects\ml-1m\users.dat', sep='::',
                    header=None, encoding='ISO-8859-1', engine='python')
users.columns = ['user_id', 'Gender', 'Age', 'OccupationID', 'ZipCode']

# Merge ratings with movies on item_id and MovieID
ratings_movies = pd.merge(ratings, movies, left_on='item_id', right_on='MovieID')
full_data = pd.merge(ratings_movies, users, left_on='user_id', right_on='user_id')

# Prepare data for recommendations
columns_to_use = ['user_id', 'item_id', 'title', 'rating']
data = full_data[columns_to_use]

# Create a pivot table
pivot_table = data.pivot_table(index='user_id', values='rating', columns='title').fillna(0)

# Calculate Item Similarity
from sklearn.metrics.pairwise import cosine_similarity
item_similarity = cosine_similarity(pivot_table.T)
item_similarity_df = pd.DataFrame(item_similarity, index=pivot_table.columns, columns=pivot_table.columns)

# Calculate User Similarity
user_similarity = cosine_similarity(pivot_table)
user_similarity_df = pd.DataFrame(user_similarity, index=pivot_table.index, columns=pivot_table.index)

# Generate Hybrid Recommendations
def get_hybrid_recommendations(user_id, num_recommendations=5):
    # Get user-based recommendations
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    similar_users_ratings = pivot_table.loc[similar_users.index]
    user_weighted_ratings = similar_users_ratings.T.dot(similar_users)

    # Get item-based recommendations
    user_ratings = pivot_table.loc[user_id]
    item_based_recommendations = item_similarity_df.dot(user_ratings).sort_values(ascending=False)

    # Combine recommendations
    combined_recommendations = user_weighted_ratings.add(item_based_recommendations, fill_value=0)

    # Remove movies already rated by the user
    combined_recommendations = combined_recommendations[user_ratings == 0]

    # Get the top recommendations
    final_recommendations = combined_recommendations.sort_values(ascending=False).head(num_recommendations)

    return final_recommendations.index.tolist()



# Define a route for home
@app.route('/')
def home():
    return render_template('index.html')  # Your form page

# Define a route for recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])  # Convert to integer
    if user_id not in pivot_table.index:
        return render_template('recommendations.html', recommendations=[], error="User ID not found.")
    recommendations = get_hybrid_recommendations(user_id)
    return render_template('recommendations.html', recommendations=recommendations)

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=5001)

