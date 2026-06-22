import streamlit as st
import numpy as np
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
session = ort.InferenceSession('model.onnx')
input_name = session.get_inputs()[0].name  # 'dense_input'

# Load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit App
st.title('Customer Churn Prediction')

# User Inputs
geography = st.selectbox(
    'Geography',
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    'Gender',
    label_encoder_gender.classes_
)

age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=900)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)

tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)

has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Create input dataframe
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'Geography': [geography]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform(
    input_data[['Geography']]
)

geo_encoded_df = pd.DataFrame(
    geo_encoded.toarray() if hasattr(geo_encoded, "toarray") else geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out()
)

# Drop Geography and combine encoded columns
input_data = pd.concat(
    [input_data.drop('Geography', axis=1), geo_encoded_df],
    axis=1
)

# Reorder columns to match training data
if hasattr(scaler, 'feature_names_in_'):
    input_data = input_data[scaler.feature_names_in_]

# Scale input
input_data_scaled = scaler.transform(input_data)

# Prediction
input_array = input_data_scaled.astype(np.float32)
prediction = session.run(None, {input_name: input_array})
prediction_proba = prediction[0][0][0]

st.subheader("Prediction Result")

if prediction_proba > 0.5:
    st.error(
        f'The customer is likely to churn. Probability: {prediction_proba:.2%}'
    )
else:
    st.success(
        f'The customer is not likely to churn. Probability: {prediction_proba:.2%}'
    )