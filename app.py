from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("kmeans_spotify.pkl")
scaler = joblib.load("scaler_spotify.pkl")

# Load dataset
df = pd.read_csv("SpotifyFeatures.csv")

# Select audio features
features = ['danceability', 'energy', 'valence', 'tempo', 'loudness',
            'acousticness', 'instrumentalness', 'liveness', 'speechiness']

# Prepare scaled data
data = df[features].dropna()
df['Cluster'] = model.predict(scaler.transform(data)).astype(str)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    song_name = request.form['song_name']

    # Find song by name
    song_row = df[df['track_name'].str.lower() == song_name.lower()]
    if song_row.empty:
        return render_template('index.html', prediction_text="❌ Song not found.")

    # Get song features and cluster
    input_features = song_row[features]
    scaled_input = scaler.transform(input_features)
    cluster = model.predict(scaled_input)[0]

    # Recommend 5 random songs from same cluster
    recommendations = df[df['Cluster'] == str(cluster)].sample(5)[['track_name', 'artist_name', 'genre']]

    return render_template('index.html',
                           prediction_text=f"🎧 Song belongs to Cluster: {cluster}",
                           recommendations=recommendations.to_html(classes="table table-striped", index=False))

if __name__ == '__main__':
    app.run(debug=True)