import streamlit as st
import pandas as pd
import joblib

# Load trained Random Forest model
rf_model = joblib.load("income_model.pkl")

st.title("Income Prediction App")
st.write("Predict whether a person earns **>50K or <=50K** based on input features.")

# User Input

def user_input_features():
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    fnlwgt = st.number_input("Final Weight (fnlwgt)", value=100000)
    education_num = st.number_input("Education Number", min_value=1, max_value=16, value=10)
    capital_gain = st.number_input("Capital Gain", value=0)
    capital_loss = st.number_input("Capital Loss", value=0)
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)

    # For categorical columns, we can define possible categories manually or infer from training
    workclass = st.selectbox("Workclass", ["Private","Self-emp-not-inc","Self-emp-inc","Federal-gov",
                                           "Local-gov","State-gov","Without-pay","Never-worked"])
    education = st.selectbox("Education", ["Bachelors","HS-grad","11th","Masters","9th","Some-college",
                                           "Assoc-acdm","Assoc-voc","7th-8th","Doctorate","Prof-school",
                                           "5th-6th","10th","1st-4th","Preschool","12th"])
    marital_status = st.selectbox("Marital Status", ["Married-civ-spouse","Divorced","Never-married","Separated",
                                                     "Widowed","Married-spouse-absent","Married-AF-spouse"])
    occupation = st.selectbox("Occupation", ["Tech-support","Craft-repair","Other-service","Sales",
                                             "Exec-managerial","Prof-specialty","Handlers-cleaners",
                                             "Machine-op-inspct","Adm-clerical","Farming-fishing",
                                             "Transport-moving","Priv-house-serv","Protective-serv","Armed-Forces"])
    relationship = st.selectbox("Relationship", ["Wife","Own-child","Husband","Not-in-family","Other-relative","Unmarried"])
    race = st.selectbox("Race", ["White","Asian-Pac-Islander","Amer-Indian-Eskimo","Other","Black"])
    sex = st.selectbox("Gender", ["Male","Female"])
    native_country = st.selectbox("Native Country", ["United-States","Cuba","Jamaica","India",
                                                     "Mexico","Puerto-Rico","Honduras","England",
                                                     "Canada","Germany","Philippines","Italy",
                                                     "Poland","Columbia","Cambodia","Thailand",
                                                     "Ecuador","Laos","Taiwan","Haiti","Portugal",
                                                     "Dominican-Republic","El-Salvador","France",
                                                     "Guatemala","China","Japan","Yugoslavia",
                                                     "Peru","Outlying-US(Guam-USVI-etc)","Scotland",
                                                     "Trinadad&Tobago","Greece","Nicaragua","Vietnam",
                                                     "Hong","Ireland","Hungary","Holand-Netherlands"])

    data = {
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education.num": education_num,
        "marital.status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital.gain": capital_gain,
        "capital.loss": capital_loss,
        "hours.per.week": hours_per_week,
        "native.country": native_country
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Preprocessing (One-hot Encode)

input_encoded = pd.get_dummies(input_df)

# Align input with model's training columns
train_columns = rf_model.feature_names_in_
input_encoded = input_encoded.reindex(columns=train_columns, fill_value=0)

# Prediction
prediction = rf_model.predict(input_encoded)
prediction_proba = rf_model.predict_proba(input_encoded)

st.subheader("Prediction")
st.write("Income >50K" if prediction[0]==1 else "Income <=50K")

st.subheader("Prediction Probability")
st.write(prediction_proba)
