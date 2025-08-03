
import pandas as pd
import numpy as np
import streamlit as st
import pickle
from PIL import Image

# page configuration
st.set_page_config(page_title= 'Disease Prediction', layout='wide')

#sidebar navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio("Go To",['Home', 'Parkinsons', 'Kidney', 'Liver'])

#Home page
if page == 'Home':
    st.title('Disease Prediction')
    st.write(""" Welcome to the Disease Prediction System.
             This application utilizes machine learning models to assist in the early detection of medical conditions such as 
             Parkinson’s disease, liver disorders, and kidney ailments. By entering relevant clinical data, users can obtain predictive
              insights designed to support informed decision-making in healthcare settings.""")   

    #loading images
    image1 = Image.open('img2.webp')
    image2 = Image.open('img3.webp')

    # Display side-by-side
    col1, col2 = st.columns(2)

    with col1:
        st.image(image1, use_column_width=True)

    with col2:
        st.image(image2, use_column_width=True)
        
# Parkinsons page

elif page == 'Parkinsons':

    st.title("Parkinson's Disease Data Input Form")

    # Feature names based on your model
    feature_names = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
        "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
        "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
        "status", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]

    with st.form("parkinsons_form"):
        st.subheader("Enter values for each feature")

        cols = st.columns(3)  # Create 3 columns
        inputs = []

        for i, feature in enumerate(feature_names):
            col = cols[i % 3]
            with col:
                value = st.number_input(feature, key=feature)
                inputs.append(value)

        submitted = st.form_submit_button("Submit")

        if submitted:
            df = pd.DataFrame([inputs], columns=feature_names)
            st.success("Form submitted successfully!")
            st.write("Your Input:")
            st.dataframe(df)

            # ⬇️ Model Loading and Prediction
            with open('parkinsons_model.pkl', 'rb') as f:
                model = pickle.load(f)

            # Drop 'status' if it was used in training as target
            if 'status' in df.columns:
                df = df.drop(columns='status')

            # Make prediction
            prediction = model.predict(df)[0]

            st.subheader("Prediction Result:")
            st.write("Parkinson's Detected ✅" if prediction == 1 else "No Parkinson's ❌")



# Kidney
elif page == 'Kidney':
    st.title("Kidney Disease Data Input Form")

    # Feature types
    categorical_features = {
        'rbc': ['normal', 'abnormal'],
        'pc': ['normal', 'abnormal'],
        'pcc': ['present', 'notpresent'],
        'ba': ['present', 'notpresent'],
        'htn': ['yes', 'no'],
        'dm': ['yes', 'no'],
        'cad': ['yes', 'no'],
        'appet': ['good', 'poor'],
        'pe': ['yes', 'no'],
        'ane': ['yes', 'no']
    }

    numerical_features = [
        'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot',
        'hemo', 'pcv', 'wc', 'rc'
    ]

    all_features = numerical_features + list(categorical_features.keys())
    feature_names = all_features

    with st.form("kidney_form"):
        st.subheader("Enter Patient Data")

        inputs = {}
        cols = st.columns(3)

        for i, feature in enumerate(all_features):
            col = cols[i % 3]
            with col:
                if feature in numerical_features:
                    inputs[feature] = st.number_input(feature, format="%.3f", key=feature)
                else:
                    inputs[feature] = st.selectbox(feature, categorical_features[feature], key=feature)

        submitted = st.form_submit_button("Submit")

        if submitted:
            # ➤ Map categorical values to numeric codes
            categorical_mappings = {
                'rbc': {'normal': 0, 'abnormal': 1},
                'pc': {'normal': 0, 'abnormal': 1},
                'pcc': {'notpresent': 0, 'present': 1},
                'ba': {'notpresent': 0, 'present': 1},
                'htn': {'no': 0, 'yes': 1},
                'dm': {'no': 0, 'yes': 1},
                'cad': {'no': 0, 'yes': 1},
                'appet': {'poor': 0, 'good': 1},
                'pe': {'no': 0, 'yes': 1},
                'ane': {'no': 0, 'yes': 1}
            }

            for feature, mapping in categorical_mappings.items():
                inputs[feature] = mapping[inputs[feature]]

            df = pd.DataFrame([inputs])
            st.success("Form submitted successfully!")
            st.write("Encoded Input:")
            st.dataframe(df)

            with open('kidney_model.pkl', 'rb') as f:
                kidney_model = pickle.load(f)

            expected_features = list(kidney_model.feature_names_in_)


            # After form submission and encoding:
            X = pd.DataFrame([inputs])[expected_features]
            prediction = kidney_model.predict(X)[0]
            probability = kidney_model.predict_proba(X)[0][1]

            # Human-readable text output
            if prediction == 1:
                st.subheader("🩺 Prediction Result:")
                st.markdown("### ⚠️ The model predicts that the patient **has kidney disease**.")
            else:
                st.subheader("🩺 Prediction Result:")
                st.markdown("### ✅ The model predicts that the patient **does not have kidney disease**.")

                st.write(f"🧪 Confidence Score: **{probability:.2%}**")


#Liver
elif page == 'Liver':
    st.title('Liver Disease Data Input Form')

     # Load model, scaler, and feature names
    with open('liver_model.pkl', 'rb') as f:
        liver_model = pickle.load(f)
    if not hasattr(liver_model, "monotonic_cst"):
        liver_model.monotonic_cst = None

    with open('liver_scaler.pkl', 'rb') as f:
        liver_scaler = pickle.load(f)
    with open('liver_features.pkl', 'rb') as f:
        liver_features = pickle.load(f) 


    #feature names
    categorical_features = {'gender' : ['male', 'female']}

    numerical_features = ['age', 'total_bilirubin', 'direct_bilirubin',
       'alkaline_phosphotase', 'alamine_aminotransferase',
       'aspartate_aminotransferase', 'total_protiens', 'albumin',
       'albumin_and_globulin_ratio']
    
    all_features = numerical_features + list(categorical_features.keys())
    feature_names = all_features

    with st.form("Liver_form"):
        st.subheader("Enter Patient Data")

        inputs = {}
        cols = st.columns(2)

        for i, feature in enumerate(liver_features):
            col = cols[i % 2]
            with col:
                if feature == 'gender':
                    inputs[feature] = st.selectbox(feature, ['male', 'female'])
                else:
                    inputs[feature] = st.number_input(feature, format="%.2f", key=feature)

        submitted = st.form_submit_button('Submit')

    if submitted:
        # Encode gender
        inputs['gender'] = 1 if inputs['gender'] == 'male' else 0

        # Create full input DataFrame
        input_df = pd.DataFrame([inputs])

        # Pass only the columns that the scaler was trained with
        scaler_input_cols = liver_scaler.feature_names_in_
        df_scaled = liver_scaler.transform(input_df[scaler_input_cols])

        # Combine scaled values and gender (if model was trained with it)
                # Combine scaled values with gender
        X_scaled_df = pd.DataFrame(df_scaled, columns=scaler_input_cols)
        X_scaled_df['gender'] = input_df['gender'].values

        # Reorder columns to match training
        X_final = X_scaled_df[liver_features]  # this ensures correct order

        # Predict
        prediction = liver_model.predict(X_final)[0]

        prob = liver_model.predict_proba(X_final)[0][1]

        # Display result
        st.subheader("🩺 Prediction Result:")
        if prediction == 1:
            st.markdown("### ⚠️ The model predicts that the patient **has liver disease**.")
        else:
            st.markdown("### ✅ The model predicts that the patient **does not have liver disease**.")
        #st.write(f"🧪 Confidence Score: **{prob:.2%}**")


        






