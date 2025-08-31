import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model
try:
    # Assuming the model is saved in the current directory or a known path
    model_path = hf_hub_download(repo_id="pawanmall/Visit-with-us", filename="best_tourism_model_v1.joblib")
    model = joblib.load(model_path)

    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    
st.title("Wellness Tourism Package Purchase Prediction")

st.write("Enter the customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

# Define input fields based on the features used in the model
# Numeric features: 'Age', 'NumberOfPersonVisiting', 'NumberOfTrips', 'NumberOfChildrenVisiting', 'MonthlyIncome', 'PitchSatisfactionScore', 'NumberOfFollowups', 'DurationOfPitch'
# Categorical features: 'TypeofContact', 'CityTier', 'Occupation', 'Gender', 'PreferredPropertyStar', 'MaritalStatus', 'Designation', 'Passport', 'OwnCar'

age = st.slider("Age", 18, 80, 30)
number_of_person_visiting = st.slider("Number of Persons Visiting", 1, 10, 1)
number_of_trips = st.slider("Number of Trips Annually", 0, 50, 5)
number_of_children_visiting = st.slider("Number of Children Visiting (under 5)", 0, 5, 0)
monthly_income = st.number_input("Monthly Income", 10000, 500000, 50000)
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
number_of_followups = st.slider("Number of Followups", 0, 10, 3)
duration_of_pitch = st.slider("Duration of Pitch (minutes)", 1, 60, 15)
passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

type_of_contact = st.selectbox("Type of Contact", ['Company Invited', 'Self Inquiry'])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer', 'Government Sector', 'Retired', 'Student'])
gender = st.selectbox("Gender", ['Male', 'Female'])
preferred_property_star = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP', 'Senior Executive', 'Junior Executive', 'Director', 'Assistant Manager', 'Lead'])
product_pitched = st.selectbox("Product Pitched", ['Destination', 'Resort', 'Cruise', 'Holiday', 'Accommodation', 'Flight', 'Walk in'])


if st.button("Predict Purchase"):
    # Create a DataFrame from the input values
    input_data = {
        'Age': [age],
        'NumberOfPersonVisiting': [number_of_person_visiting],
        'NumberOfTrips': [number_of_trips],
        'NumberOfChildrenVisiting': [number_of_children_visiting],
        'MonthlyIncome': [monthly_income],
        'PitchSatisfactionScore': [pitch_satisfaction_score],
        'NumberOfFollowups': [number_of_followups],
        'DurationOfPitch': [duration_of_pitch],
        'Passport': [passport],
        'OwnCar': [own_car],
        'TypeofContact': [type_of_contact],
        'CityTier': [city_tier],
        'Occupation': [occupation],
        'Gender': [gender],
        'PreferredPropertyStar': [preferred_property_star],
        'MaritalStatus': [marital_status],
        'Designation': [designation],
        'ProductPitched': [product_pitched] # Include ProductPitched for prediction
    }
    input_df = pd.DataFrame(input_data)

    try:
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[:, 1] # Probability of purchasing

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.success(f"The customer is likely to purchase the package.")
        else:
            st.warning(f"The customer is unlikely to purchase the package.")

        st.write(f"Probability of Purchase: {prediction_proba[0]:.2f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
