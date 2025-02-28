import numpy as np 
import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

def load_model():
    with  open('Diabetes_predict.pkl','rb') as file:
      model,scaler=pickle.load(file)
      return model,scaler
    
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
def load_model(model_name):
   
    with open(model_name, 'rb') as file:
        model, scaler = pickle.load(file)
    return model, scaler
def processing_input_data(data, scaler):
    
    data = pd.DataFrame([data])
    data["SEX"] = data["SEX"].map({"Male": 1, "Female": 2})
    data_transformed = scaler.transform(data)
    return data_transformed

def predict_data(data, model_name):

    model, scaler = load_model(model_name)
    processed_data = processing_input_data(data, scaler)
    prediction = model.predict(processed_data)
    return prediction

df['target']  = diabetes.target

x = df.drop(columns=['target'])
y = df['target']


x_train , x_test , y_train,y_test = train_test_split(x,y , test_size=.2)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

ridge_model = Ridge()
lasso_model = Lasso()

ridge_model.fit(x_train_scaled,y_train)

lasso_model.fit(x_train_scaled,y_train)

ridge_model.predict(x_test_scaled)
lasso_model.predict(x_test_scaled)

mean_squared_error(y_test,ridge_model.predict(x_test_scaled))
mean_squared_error(y_test,lasso_model.predict(x_test_scaled))

st.title("Diabetes prediction")
st.write("Enter your Name")
    

    
if st.button("predict-your_score"):
        Age = st.number_input("Age",min_value = 1, max_value = 100 , value = 20)
        Sex = st.number_input("Sex", min_value = 0, max_value = 1 , value = 0)
        Bmi = st.number_input("BMI", min_value = 1, max_value = 100, value = 10)
        BP = st.number_input("BP", min_value = 1, max_value = 100, value = 10)
        s1 = st.number_input("s1", min_value = 1, max_value = 100, value = 10)
        s2 = st.number_input("s2", min_value = 1, max_value = 100, value = 10)
        s3 = st.number_input("s3", min_value = 1, max_value = 100, value = 10)
        s4 = st.number_input("s4", min_value = 1, max_value = 100, value = 10)
        s5 = st.number_input("s5", min_value = 1, max_value = 100, value = 10)
        s6 = st.number_input("s6", min_value = 0, max_value = 100, value = 10)
        user_data = {
                "AGE": Age,
                "SEX": Sex,
                "BMI": Bmi,
                "BP": BP,
                "S1": s1,
                "S2": s2,
                "S3": s3,
                "S4": s4,
                "S5": s5,
                "S6": s6
            }

prediction = predict_data(user_data)
st.success(f"your prediciotn result is {prediction}")
user_data["prediction"] = round(float (prediction[0]),2)
user_data = {key: int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, np.floating) else value for key, value in user_data.items()}
