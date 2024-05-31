# -*- coding: utf-8 -*-
import sys
import os
import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import streamlit.components.v1 as components

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'UTF-8'

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

# Function to classify an image
def classify(image, model, class_names):
    # Convert image to (150, 150)
    image = ImageOps.fit(image, (150, 150), Image.Resampling.LANCZOS)
    
    # Convert image to numpy array
    image_array = np.asarray(image)
    
    # Normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Set model input
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    return class_name, confidence_score

def main_page():
    set_background('stethoscope-2617701.jpg')
    
    # Define the HTML for the header
    header_html = """
    <h1 style="font-size: 100px; color: white; text-align: left;">
        <b>LABORATORY</b><br>
        <b>AND</b><br>
        <b>RADIOLOGY</b>
    </h1>
    """

    # Set background color to make the white text visible (optional)
    page_bg_style = """
    <style>
        body {
            background-color: black;
        }
    </style>
    """

    st.markdown(header_html, unsafe_allow_html=True)
    st.write('_YOU ASK AND WE PREDICT_')
    

def Laboratory_Department_page():
    set_background('stethoscope-2617701.jpg')
    header_html = """
    <h1 style="font-size: 80px; color: white; text-align: left;">
        <b>LABORATORY</b><br>
        <b>Department</b>
    </h1>
    """

    # Set background color to make the white text visible (optional)
    page_bg_style = """
    <style>
        body {
            background-color: black;
        }
    </style>
    """

    st.markdown(header_html, unsafe_allow_html=True)
    
    st.write("In this sction you can know what is your body analysis results refer to ,")
    st.write("all of diseases have specific body analysis results and some hapits to pridect well")
    
    header_html = """
    <h1 style="font-size: 40px; color: white; text-align: left;">
        <b>Choose from the following  diseases:</b>
    </h1>
    """

    st.markdown(header_html, unsafe_allow_html=True)
    
    
    ##st.header("_choose the disease:_")
    

   
    disease = st.radio(
        'Diseases',
        ["***Obese***", "***Eye_vision***", "***blood_pressure***", "***Heart_health***",
         "***Anemia***", "***s_creatinine_range***", "***liver_enzymes***"],
        index=None,
        captions=["you must first calculate BMI() = (Hight*Hight)/Weight"]
    )

    if disease == "***Obese***":
        
        st.write("You selected Obese.")
        st.write('Answer the following questions.')

        # Load the model
        model = joblib.load('Obese_model.pkl')

        # Prediction function
        def predict_obese(BMI, waistline, age, sex):
            prediction = model.predict_proba(pd.DataFrame({
                'sex': [sex],
                'waistline': [waistline],
                'age': [age],
                'BMI': [BMI]
            }))

            label = ['thinny', 'normal', 'obese', 'overly fat']
            predicted_class_index = prediction.argmax()
            return label[predicted_class_index]

        # Input fields
        BMI = st.text_input('BMI', 'Please enter your BMI')
        sex = st.radio('Pick your gender', ['Male', 'Female'])
        age = st.text_input('Age', 'Please enter your age')
        waistline = st.text_input('Waistline', 'Please enter your waistline')

        # Prediction result
        result = ""

        if st.button('Predict'):
            result = predict_obese(BMI, waistline, age, sex)
            st.success(f'You have {result}')
            
            
    if  disease == "***Eye_vision***":
        st.write("You selected Eye_vision.")
        st.write('Answer the following questions.')

        # Load the model
        model = joblib.load('Eye_vision_model.pkl')

        # Prediction function
        def predict_Eye_vision(sight_right, sight_left,  hemoglobin, age, sex):
            prediction = model.predict_proba(pd.DataFrame({
                'sex': [sex],
                'sight_right': [sight_right],
                'age': [age],
                'sight_left': [sight_left],
                'hemoglobin' : [hemoglobin]
            }))

            label = ['poor vision' , 'normal vision']
            predicted_class_index = prediction.argmax()
            return label[predicted_class_index]

        # Input fields
        sight_left = st.text_input('sight_left', 'Please enter your sight_left')
        sight_right = st.text_input('sight_right', 'Please enter your sight_right')
        sex = st.radio('Pick your gender', ['Male', 'Female'])
        age = st.text_input('Age', 'Please enter your age')
        hemoglobin = st.text_input('hemoglobin', 'Please enter your hemoglobin')

        # Prediction result
        result = ""

        if st.button('Predict'):
            result = predict_Eye_vision(sight_right, sight_left,  hemoglobin, age, sex)
            st.success(f'You have {result}')
            
            
            
    if  disease == "***blood_pressure***":
        st.write("You selected blood_pressure .")
        st.write('Answer the following questions.')

        # Load the model
        model = joblib.load('blood_pressure_model.pkl')

        # Prediction function
        def predict_blood_pressure(SBP, DBP,  weight, age, sex):
            prediction = model.predict_proba(pd.DataFrame({
                'sex': [sex],
                'age': [age],
                'DBP': [DBP],
                'SBP' : [SBP],
                'weight' : [weight]
            }))

            label = ['normal' , 'hypotension' , 'over Hypertension' , 'hypertension']
            predicted_class_index = prediction.argmax()
            return label[predicted_class_index]

        # Input fields
        SBP = st.text_input('SBP', 'Please enter your SBP')
        DBP = st.text_input('DBP', 'Please enter your DBP')
        sex = st.radio('Pick your gender', ['Male', 'Female'])
        age = st.text_input('Age', 'Please enter your age')
        weight = st.text_input('weight', 'Please enter your weight')

        # Prediction result
        result = ""

        if st.button('Predict'):
            result = predict_blood_pressure(SBP, DBP,  weight, age, sex)
            st.success(f'You have {result}')
            
            
            
    if  disease == "***Heart_health***":
        st.write("You selected Heart_health .")
        st.write('Answer the following questions.')

        # Load the model
        model = joblib.load('heart_health_model.pkl')

        # Prediction function
        def predict_Heart_health(LDL_chole , HDL_chole , tot_chole , triglyceride ,  weight , age , sex):
            prediction = model.predict_proba(pd.DataFrame({
                'sex': [sex],
                'age': [age],
                'LDL_chole': [LDL_chole],
                'HDL_chole' : [HDL_chole],
                'tot_chole' : [tot_chole],
                'triglyceride' : [triglyceride],
                'weight' : [weight]
            }))

            label = ['At_risk' , 'Heart_healthy' , 'Dangerous']
            predicted_class_index = prediction.argmax()
            return label[predicted_class_index]

        # Input fields
        LDL_chole = st.text_input('LDL_chole', 'Please enter your LDL_chole')
        HDL_chole = st.text_input('DBP', 'Please enter your HDL_chole')
        tot_chole = st.text_input('tot_chole', 'Please enter your tot_chole')
        triglyceride = st.text_input('triglyceride', 'Please enter your triglyceride')
        sex = st.radio('Pick your gender', ['Male', 'Female'])
        age = st.text_input('Age', 'Please enter your age')
        weight = st.text_input('weight', 'Please enter your weight')

        # Prediction result
        result = ""

        if st.button('Predict'):
            result = predict_Heart_health(LDL_chole , HDL_chole , tot_chole , triglyceride ,  weight , age , sex)
            st.success(f'You have {result}')
            
            
    if  disease == "***Anemia***":
        st.write("You selected Anemia .")
        st.write('Answer the following questions.')

        # Load the model
        model = joblib.load('anemia_model.pkl')

        # Prediction function
        def predict_Anemia(hemoglobin , height , age , sex):
            prediction = model.predict_proba(pd.DataFrame({
                'sex': [sex],
                'age': [age],
                'hemoglobin': [hemoglobin],
                'height' : [height]
            }))

            label = ['Normal' , 'Anemia']
            predicted_class_index = prediction.argmax()
            return label[predicted_class_index]

        # Input fields
        sex = st.radio('Pick your gender', ['Male', 'Female'])
        age = st.text_input('Age', 'Please enter your age')
        hemoglobin = st.text_input('hemoglobin', 'Please enter your hemoglobin')
        height = st.text_input('height', 'Please enter your height')

        # Prediction result
        result = ""

        if st.button('Predict'):
            result = predict_Anemia(hemoglobin , height , age , sex)
            st.success(f'You have {result}')
            
            
            
            
    if  disease == "***s_creatinine_range***":
        st.write("You selected s_creatinine_range .")
        st.write('Answer the following questions.')

        # Load the model
        model = joblib.load('s_creatinine_range_model.pkl')

        # Prediction function
        def predict_s_creatinine_range(serum_creatinine , hemoglobin ,  height , sex , age
                                       ):
            prediction = model.predict_proba(pd.DataFrame({
                'sex': [sex],
                'age' : [age],
                'serum_creatinine' : [serum_creatinine],
                'height' : [height],
                'hemoglobin' : [hemoglobin]
                
            }))

            label = ['Average' , 'Normal' , 'High']
            predicted_class_index = prediction.argmax()
            return label[predicted_class_index]

        # Input fields
        sex = st.radio('Pick your gender', ['Male', 'Female'])
        age = st.text_input('age', 'Please enter your age')
        serum_creatinine = st.text_input('serum_creatinine', 'Please enter your serum_creatinine')
        height = st.text_input('height', 'Please enter your height')
        hemoglobin = st.text_input('hemoglobin', 'Please enter your hemoglobin')




        # Prediction result
        result = ""

        if st.button('Predict'):
            result = predict_s_creatinine_range(serum_creatinine , hemoglobin ,  height , sex , age
                                               )
            st.success(f'You have {result}')
            
            
    if  disease == "***liver_enzymes***":
        st.write("You selected liver_enzymes .")
        st.write('Answer the following questions.')

        # Load the model
        model = joblib.load('liver_enzymes_model.pkl')

        # Prediction function
        def predict_liver_enzymes(gamma_GTP, SGOT_ALT, SGOT_AST, triglyceride, DRK_YN, sex , weight ,
                                                SMK_stat_type_cd
                                       ):
            prediction = model.predict_proba(pd.DataFrame({
                'sex': [sex],
                'gamma_GTP': [gamma_GTP],
                'SGOT_ALT': [SGOT_ALT],
                'SGOT_AST' : [SGOT_AST],
                'triglyceride' : [triglyceride],
                'weight' : [weight],
                'SMK_stat_type_cd' : [SMK_stat_type_cd],
                'DRK_YN' : [DRK_YN]
                
            }))

            label = ['Normal' 'High' 'Very High']
            predicted_class_index = prediction.argmax()
            return label[predicted_class_index]

        # Input fields
        sex = st.radio('Pick your gender', ['Male', 'Female'])
        gamma_GTP = st.text_input('gamma_GTP', 'Please enter your gamma_GTP')
        SGOT_ALT = st.text_input('SGOT_ALT', 'Please enter your SGOT_ALT')
        SGOT_AST = st.text_input('SGOT_AST', 'Please enter your SGOT_AST')
        triglyceride = st.text_input('triglyceride', 'Please enter your triglyceride')
        weight = st.text_input('weight', 'Please enter your weight')
        SMK_stat_type_cd = st.text_input('SMK_stat_type_cd', 'Please enter your SMK_stat_type_cd')
        DRK_YN = st.text_input('DRK_YN', 'Please enter your DRK_YN')




        # Prediction result
        result = ""

        if st.button('Predict'):
            result = predict_liver_enzymes(gamma_GTP, SGOT_ALT, SGOT_AST, triglyceride, DRK_YN, sex , weight ,
                                                SMK_stat_type_cd
                                               )
            st.success(f'You have {result}')
            
    else:
        st.write("You didn't select a disease.")


def Radiology_Department_page():
    set_background('stethoscope-2617701.jpg')
    header_html = """
    <h1 style="font-size: 80px; color: white; text-align: left;">
        <b>Radiology</b><br>
        <b>Department</b>
    </h1>
    """

    # Set background color to make the white text visible (optional)
    page_bg_style = """
        <style>
            body {
                background-color: black;
            }
        </style>
    """

    st.markdown(header_html, unsafe_allow_html=True)
    st.write("In this sction you can know what is your X-ray results refer to ,")
    st.write("just upload the X-ray image and the result will be under it")
    
    header_html = """
    <h1 style="font-size: 40px; color: white; text-align: left;">
        <b>choose from the following  diseases:</b>
    </h1>
    """

    st.markdown(header_html, unsafe_allow_html=True)
    
   
    
    disease = st.radio(
        "Diseases",
        ["***Brian_tumer***", "***Lungs_radiology***", "***Skin_cancer***"
         ],
        index=None,
        captions=["A brain tumor can form in the brain cells (as shown), or it can begin elsewhere and spread to the brain",
                 "The lungs are the functional units of respiration and are key to survival",
                  "Skin cancer — the abnormal growth of skin cells — most often develops on skin exposed to the sun"
                 
                 ]
    )
    
    
    if disease == "***Brian_tumer***":
        st.write("You selected Brian_tumer .")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            st.write("")
            st.write("Result...")

            class_names = ['glioma', 'Healthy', 'meningioma', 'notumor', 'pituitary', 'Testing']  # Replace with your class names
            model = load_model('Brian_tumer_model.h5')
            class_name, conf_score = classify(image, model, class_names)

            st.write(f"An image refer to: {class_name}")
            st.write(f"Confidence: {conf_score * 100:.2f}%")
        
    
    
    
    if disease == "***Lungs_radiology***":
        st.write("You selected Lungs_radiology .")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            st.write("")
            st.write("Result...")

            class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']  # Replace with your class names
            model = load_model('Lungs_radiology_model.h5')
            class_name, conf_score = classify(image, model, class_names)

            st.write(f"An image refer to: {class_name}")
            st.write(f"Confidence: {conf_score * 100:.2f}%")
            
            
    if disease == "***Skin_cancer***":
        st.write("You selected Skin_cancer .")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            st.write("")
            st.write("Result...")

            class_names = ['actinic keratosis', 'basal cell carcinoma',
                           'dermatofibroma', 'melanoma', 'nevus', 'pigmented benign keratosis', 
                           'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']  
            model = load_model('Skin_cancer_model.h5')
            class_name, conf_score = classify(image, model, class_names)

            st.write(f"An image refer to: {class_name}")
            st.write(f"Confidence: {conf_score * 100:.2f}%")
    

def about_page():
    st.title("About")
    st.write("This is an app for labor results and radiology.")



# Define the pages in a dictionary
pages = {
    "Main Page": main_page,
    "Radiology Department": Radiology_Department_page,
    "Laboratory Department": Laboratory_Department_page,
    "About": about_page
}

# Create a select box in the sidebar for navigation
st.sidebar.header("**Available Services**")
selection = st.sidebar.selectbox("Go to", list(pages.keys()))

# Display the selected page
page = pages[selection]
page()
