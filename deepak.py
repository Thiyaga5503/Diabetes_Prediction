import streamlit as st
import pickle
import numpy as np
from pymongo import MongoClient

# Load the trained model and scaler
with open("lasso_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Connect to MongoDB
client = MongoClient(st.secrets["MONGO_URI"])
db = client["diabetes_db"]
collection = db["predictions"]

# Define feature names
feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

st.title("Diabetes Progression Prediction")
st.write("Enter the values for each feature to predict the target.")

# User name input
user_name = st.text_input("Enter your name")

# Arrange input fields into two columns
col1, col2 = st.columns(2)

inputs = []
for i, feature in enumerate(feature_names):
    if i % 2 == 0:
        value = col1.number_input(f"{feature}", value=0.0)
    else:
        value = col2.number_input(f"{feature}", value=0.0)
    inputs.append(value)

# Convert inputs to numpy array and reshape for scaling
input_array = np.array(inputs).reshape(1, -1)
scaled_input = scaler.transform(input_array)

# Predict button
if st.button("Predict"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"Predicted Target Value: {prediction:.2f}")
    
    # Store response in MongoDB
    if user_name:
        record = {"name": user_name, "features": dict(zip(feature_names, inputs)), "prediction": prediction}
        collection.insert_one(record)
        st.write("Response saved to database!")
    else:
        st.warning("Please enter your name before submitting.")

# Credit
st.write("Built by Deepak")