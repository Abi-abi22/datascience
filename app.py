from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

app = Flask(__name__)

# Load your dataset
data = pd.read_csv('gym recommendation.csv')
data.drop(columns=['ID'], inplace=True)

# Label Encoding
label_enc = LabelEncoder()
for col in ['Sex', 'Hypertension', 'Diabetes', 'Level', 'Fitness Goal', 'Fitness Type']:
    data[col] = label_enc.fit_transform(data[col])

# Standardize numeric columns
scaler = StandardScaler()
data[['Age', 'Height', 'Weight', 'BMI']] = scaler.fit_transform(data[['Age', 'Height', 'Weight', 'BMI']])

@app.route('/', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        # Get form inputs
        user_input = {
            'Sex': int(request.form['Sex']),
            'Age': float(request.form['Age']),
            'Height': float(request.form['Height']),
            'Weight': float(request.form['Weight']),
            'Hypertension': int(request.form['Hypertension']),
            'Diabetes': int(request.form['Diabetes']),
            'BMI': float(request.form['BMI']),
            'Level': int(request.form['Level']),
            'Fitness Goal': int(request.form['Fitness_Goal']),
            'Fitness Type': int(request.form['Fitness_Type'])
        }

        # Normalize user input
        num_features = ['Age', 'Height', 'Weight', 'BMI']
        user_df = pd.DataFrame([user_input], columns=num_features)
        user_df[num_features] = scaler.transform(user_df[num_features])
        user_input.update(user_df.iloc[0].to_dict())
        user_df = pd.DataFrame([user_input])

        user_features = data[['Sex', 'Age', 'Height', 'Weight', 'Hypertension', 'Diabetes', 'BMI', 'Level', 'Fitness Goal', 'Fitness Type']]
        similarity_scores = cosine_similarity(user_features, user_df).flatten()
        similar_user_indices = similarity_scores.argsort()[-5:][::-1]
        similar_users = data.iloc[similar_user_indices]
        recommendation = similar_users[['Exercises', 'Diet', 'Equipment']].mode().iloc[0]

        return f"""
            <h2>Recommended Plan</h2>
            <p><strong>Exercises:</strong> {recommendation['Exercises']}</p>
            <p><strong>Diet:</strong> {recommendation['Diet']}</p>
            <p><strong>Equipment:</strong> {recommendation['Equipment']}</p>
        """

    return render_template('form.html')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
