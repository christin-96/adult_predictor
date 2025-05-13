# Deploy adult predictor

# ==========================================================================
import pandas as pd
import numpy as np

import streamlit as st
import pickle

# ==========================================================================
# Main Title
st.write("""ADULT INCOME PREDICTOR""")
st.write("""Predict if a person makes more than $50,000 a year""")

st.sidebar.header('User Input Features')

# ==========================================

# Create function to get user input
def user_input_features():
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=25)
    workclass = st.sidebar.selectbox("Workclass", ("Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                                                    "Local-gov", "State-gov", "Without-pay", "Never-worked"))
    education_num = st.sidebar.slider("Education Num", min_value=1, max_value=16, value=10)
    marital_status = st.sidebar.selectbox("Marital Status", ("Married-civ-spouse", "Divorced", "Never-married",
                                                               "Separated", "Widowed", "Married-spouse-absent",
                                                               "Married-AF-spouse"))
    occupation = st.sidebar.selectbox("Occupation", ("Tech-support", "Craft-repair", "Other-service",
                                                      "Sales", "Exec-managerial", "Prof-specialty",
                                                      "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                                                      "Farming-fishing", "Transport-moving", "Priv-house-serv",
                                                      "Protective-serv", "Armed-Forces"))
    relationship = st.sidebar.selectbox("Relationship", ("Wife", "Own-child", "Husband", "Not-in-family",
                                                           "Other-relative", "Unmarried"))
    race = st.sidebar.selectbox("Race", ("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Black", "Other"))
    sex = st.sidebar.selectbox("Sex", ("Female", "Male"))
    capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=99999, value=0)   
    capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=99999, value=0)
    hours_per_week = st.sidebar.number_input("Hours per Week", min_value=1, max_value=99, value=40)
    native_country = st.sidebar.selectbox("Native Country", ("United-States", "Cambodia", "England", "Puerto-Rico",
                                                                "Canada", "Germany", "Outlying-US(Guam-USVI-etc)",
                                                                "India", "Japan", "Greece", "South", "China",
                                                                "Cuba", "Iran", "Honduras", "Philippines",
                                                                "Italy", "Poland", "Columbia", "Jamaica",
                                                                "Vietnam", "Mexico", "Portugal", "Ireland",
                                                                "France", "Dominican-Republic", "Laos",
                                                                "Ecuador", "Taiwan", "Haiti", "Hungary",
                                                                "Guatemala", "Nicaragua", "Scotland",
                                                                "Thailand", "Yugoslavia", "El-Salvador",
                                                                "Trinadad&Tobago", "Peru", 
                                                                "Other"))
    
    df = pd.DataFrame()
    df['Age'] = [age]
    df['Workclass'] = [workclass]
    df['Education_num'] = [education_num]
    df['Marital_Status'] = [marital_status]
    df['Occupation'] = [occupation]
    df['Relationship'] = [relationship] 
    df['Race'] = [race]
    df['Sex'] = [sex]
    df['Capital_gain'] = [capital_gain]
    df['Capital_loss'] = [capital_loss]
    df['Hours_per_week'] = [hours_per_week]
    df['Native_country'] = [native_country]
    return df

# Get user input
input_df = user_input_features()

# ===========================================
# Load the model from disk
model = pickle.load(open("Model_Final.sav", "rb"))

# Make prediction
model_prediction = model.predict(input_df)

# ==========================================

# Display 2 columns
col1, col2 = st.columns(2)

# Display user input features
with col1:
    st.subheader("User Input Features")
    st.write(input_df.transpose())

# Display prediction
with col2:
    st.subheader("Prediction")
    if model_prediction == 1:
        st.write("Income MORE than $50,000 a year")
    else:
        st.write("Income LESS than $50,000 a year")
# ==========================================
# Predict csv file
data_batch = st.file_uploader("Upload CSV file", type=["csv"])
if data_batch is not None:
    batch_df = pd.read_csv(data_batch)
    st.write(batch_df)
    batch_prediction = model.predict(batch_df)
    st.subheader("Batch Prediction")
    if batch_prediction == 1:
        st.write("Income MORE than $50,000 a year")
    else:
        st.write("Income LESS than $50,000 a year")
# ===========================================

