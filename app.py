
import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Titanic Survival Prediction")

# Input form
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 100, 25)
SibSp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
Fare = st.number_input("Fare", 0.0, 600.0, 32.0)
Embarked = st.selectbox("Embarked", ["S", "C", "Q"])

if st.button("Predict"):
    sex = 1 if Sex == "male" else 0
    embark = {"S": 0, "C": 1, "Q": 2}[Embarked]
    fare = np.log(Fare + 1)

    input_data = scaler.transform([[Pclass, sex, Age, SibSp, Parch, fare, embark]])
    result = model.predict(input_data)[0]

    st.success("Survived!" if result == 1 else "Did not survive.")
