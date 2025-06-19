import streamlit as st
import numpy as np
import joblib  # or use pickle if needed

# Load the trained model and scaler
model = joblib.load("model.pkl")        # Ensure model.pkl is in the same directory
scaler = joblib.load("scaler.pkl")      # Ensure scaler.pkl is also in the same directory

# App title
st.title("🚢 Titanic Survival Prediction App")

# Input fields
Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex_option = st.selectbox("Sex", ["male", "female"])
sex = 0 if sex_option == "male" else 1
Age = st.slider("Age", 0, 100, 25)
SibSp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
Fare = st.number_input("Fare Paid", 0.0, 500.0, 32.0)
Embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
embark = {"S": 0, "C": 1, "Q": 2}[Embarked]

# Predict button
if st.button("Predict Survival"):
    try:
        fare_log = np.log(Fare + 1)
        input_data = np.array([[Pclass, sex, Age, SibSp, Parch, fare_log, embark]])
        scaled_input = scaler.transform(input_data)
        result = model.predict(scaled_input)[0]
        st.success("✅ Survived!" if result == 1 else "❌ Did not survive.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
