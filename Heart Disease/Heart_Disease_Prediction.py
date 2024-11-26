import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from PIL import Image

st.set_page_config(page_title="Heart Disease Prediction")


@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

model_path = 'heart_disease_catboost_model.pkl'

model = load_model(model_path)


image = Image.open('images.png')
st.image(image, use_column_width=True)


st.title("Heart Disease Prediction")
st.write("Enter the features to predict if the patient has heart disease or not.")


age = st.number_input('Age', min_value=0, max_value=120)
sex = st.selectbox('Sex', ['Select an option', 'male', 'female'])
cp_type = st.selectbox('Chest Pain Type', ['Select an option', 'typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
rest_bp = st.number_input('Resting Blood Pressure', min_value=0)
cholesterol = st.number_input('Cholesterol', min_value=0)
fbs = st.selectbox('Fasting Blood Sugar', ['Select an option', 'normal', 'high'])
rest_ecg = st.selectbox('Resting ECG', ['Select an option', 'normal', 'ST-T wave abnormality', 'left ventricular hypertrophy'])
max_hr = st.number_input('Maximum Heart Rate', min_value=0)
ex_angina = st.selectbox('Exercise Induced Angina', ['Select an option', 'False', 'True'])
oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0)
st_slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Select an option', 'upsloping', 'flat', 'downsloping'])

input_data = {
    'AGE': age,
    'SEX': 1 if sex == 'male' else 0 if sex == 'female' else None,
    'CHEST PAIN TYPE': {
        'typical angina': 1,
        'atypical angina': 2,
        'non-anginal pain': 3,
        'asymptomatic': 4
    }.get(cp_type, None),
    'RESTING BP S': rest_bp,
    'CHOLESTEROL': cholesterol,
    'FASTING BLOOD SUGAR': 1 if fbs == 'high' else 0 if fbs == 'normal' else None,
    'RESTING ECG': {
        'normal': 0,
        'ST-T wave abnormality': 1,
        'left ventricular hypertrophy': 2
    }.get(rest_ecg, None),
    'MAX HEART RATE': max_hr,
    'EXERCISE ANGINA': 1 if ex_angina == 'True' else 0 if ex_angina == 'False' else None,
    'OLDPEAK': oldpeak,
    'ST SLOPE': {
        'upsloping': 1,
        'flat': 2,
        'downsloping': 3
    }.get(st_slope, None)
}

if all(value is not None for value in input_data.values()):
    input_df = pd.DataFrame([input_data])

    input_df['NEW AGE'] = pd.cut(input_df['AGE'],
                           bins=[0, 12, 19, 35, 60, 100],
                           labels=['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior'])

    input_df['NEW CHOLESTEROL'] = pd.cut(input_df['CHOLESTEROL'],
                                   bins=[0, 200, 239, 279, 500],
                                   labels=['Desirable', 'Borderline High', 'High', 'Very High'])

    input_df['NEW RESTING BP S'] = pd.cut(input_df['RESTING BP S'],
                                   bins=[0, 90, 120, 130, 140, 180, 300],
                                   labels=['Low', 'Normal', 'Elevated', 'Hypertension Stage 1', 'Hypertension Stage 2', 'Hypertensive Crisis'])

    input_df['NEW MAX HEART RATE BASED ON AGE'] = input_df["MAX HEART RATE"] - input_df["AGE"]

    input_df = pd.get_dummies(input_df, columns=['CHEST PAIN TYPE', 'RESTING ECG', 'NEW AGE', 'NEW CHOLESTEROL', 'NEW RESTING BP S'], drop_first=True)

    expected_features = model.feature_names_

    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0

    input_df = input_df[expected_features]

    scaler = StandardScaler()
    scaled_input = scaler.fit_transform(input_df)

    new_data = pd.DataFrame(scaled_input, columns=expected_features)

    if st.button('Predict'):
        prediction = model.predict(new_data)

        if prediction == 1:
            st.markdown("<h2 style='text-align: center; color: red;'>Disease</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center; color: green;'>Healthy</h2>", unsafe_allow_html=True)
else:
    st.write('Please select an option for all fields to get a prediction.')
