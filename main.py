import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random

# Load data
data = pd.read_csv("gym recommendation.csv")

# Drop unnecessary column
data.drop(columns=['ID'], inplace=True)

# Label Encoding categorical columns
label_enc = LabelEncoder()
for col in ['Sex', 'Hypertension', 'Diabetes', 'Level', 'Fitness Goal', 'Fitness Type']:
    data[col] = label_enc.fit_transform(data[col])

# Normalize numerical features
scaler = StandardScaler()
data[['Age', 'Height', 'Weight', 'BMI']] = scaler.fit_transform(data[['Age', 'Height', 'Weight', 'BMI']])

def get_recommendation(top_n=3):
    print("Please enter your details for a personalized workout and diet recommendation.")
    user_input = {
        'Sex': int(input("Enter Sex (Male : 1/Female : 0): ")),
        'Age': float(input("Enter Age: ")),
        'Height': float(input("Enter Height in meters (e.g., 1.75): ")),
        'Weight': float(input("Enter Weight in kg: ")),
        'Hypertension': int(input("Do you have Hypertension (Yes : 1/No : 0): ")),
        'Diabetes': int(input("Do you have Diabetes (Yes : 1/No : 0): ")),
        'BMI': float(input("Enter BMI: ")),
        'Level': int(input("Enter Level (Underweight : 3, Normal : 0, Overweight : 2, Obese : 1): ")),
        'Fitness Goal': int(input("Enter Fitness Goal (Weight Gain : 0, Weight Loss : 1): ")),
        'Fitness Type': int(input("Enter Fitness Type (Muscular Fitness : 1, Cardio Fitness : 0): "))
    }

    # Normalize numerical features
    num_features = ['Age', 'Height', 'Weight', 'BMI']
    user_df = pd.DataFrame([user_input], columns=num_features)
    user_df[num_features] = scaler.transform(user_df[num_features])
    user_input.update(user_df.iloc[0].to_dict())
    user_df = pd.DataFrame([user_input])

    # Calculate similarity scores for exact user input
    user_features = data[['Sex', 'Age', 'Height', 'Weight', 'Hypertension', 'Diabetes', 'BMI', 'Level', 'Fitness Goal', 'Fitness Type']]
    similarity_scores = cosine_similarity(user_features, user_df).flatten()

    # Retrieve top similar users and get the first recommendation
    similar_user_indices = similarity_scores.argsort()[-5:][::-1]
    similar_users = data.iloc[similar_user_indices]
    recommendation_1 = similar_users[['Exercises', 'Diet', 'Equipment']].mode().iloc[0]

    # Simulate two additional recommendations by modifying input values slightly
    simulated_recommendations = []

    for _ in range(2):
        modified_input = user_input.copy()
        modified_input['Age'] += random.randint(-5, 5)
        modified_input['Weight'] += random.uniform(-5, 5)
        modified_input['BMI'] += random.uniform(-1, 1)

        # Normalize modified input values
        modified_user_df = pd.DataFrame([modified_input], columns=num_features)
        modified_user_df[num_features] = scaler.transform(modified_user_df[num_features])
        modified_input.update(modified_user_df.iloc[0].to_dict())

        # Calculate similarity scores for modified input
        modified_similarity_scores = cosine_similarity(user_features, pd.DataFrame([modified_input])).flatten()
        modified_similar_user_indices = modified_similarity_scores.argsort()[-5:][::-1]
        modified_similar_users = data.iloc[modified_similar_user_indices]
        recommendation = modified_similar_users[['Exercises', 'Diet', 'Equipment']].mode().iloc[0]

        if not any(rec['Exercises'] == recommendation['Exercises'] and rec['Diet'] == recommendation['Diet'] and rec['Equipment'] == recommendation['Equipment'] for rec in simulated_recommendations):
            simulated_recommendations.append(recommendation)

    # Display all recommendations
    print("\nRecommended Workout and Diet Plans based on your input:")
    print("\nRecommendation 1 (Exact match):")
    print("EXERCISES:", recommendation_1['Exercises'])
    print("EQUIPMENTS:", recommendation_1['Equipment'])
    print("DIET:", recommendation_1['Diet'])

    for idx, rec in enumerate(simulated_recommendations, start=2):
        print(f"\nRecommendation {idx} (Slight variation):")
        print("EXERCISES:", rec['Exercises'])
        print("EQUIPMENTS:", rec['Equipment'])
        print("DIET:", rec['Diet'])

    # Collect feedback for each recommendation
    feedback_matrix = []
    for i in range(len(simulated_recommendations) + 1):
        feedback = int(input(f"Was recommendation {i+1} relevant? (Yes: 1, No: 0): "))
        feedback_matrix.append(feedback)

    # Calculate MRR
    relevant_indices = [i + 1 for i, feedback in enumerate(feedback_matrix) if feedback == 1]
    if relevant_indices:
        mrr = np.mean([1 / rank for rank in relevant_indices])
    else:
        mrr = 0.0

    print(f"\nMean Reciprocal Rank (MRR): {mrr:.2f}")

    return [recommendation_1] + simulated_recommendations

# Get and display recommendations
if __name__ == "__main__":
    recommendations = get_recommendation(top_n=3)