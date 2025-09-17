# Import required libraries
import pandas as pd
import joblib

# Load pre-trained models and scalers
# Two models are used: one for young age group (<=25) and another for the rest (>25).
# Each group also has its own scaler for feature normalization..

model_young = joblib.load('artifacts/model_young.joblib')
model_rest = joblib.load('artifacts/model_rest.joblib')
scaler_young = joblib.load('artifacts/scaler_young.joblib')
scaler_rest = joblib.load('artifacts/scaler_rest.joblib')


# Function: calculate_normalized_risk
#Converts medical history into a numerical "risk score".
#Normalizes score between 0 and 1 for model input.
def calculate_normalized_risk(medical_history):
    risk_scores = {"diabetes": 6,
                   "heart disease": 8,
                   "high blood pressure": 6,
                   "thyroid": 5,
                   "no disease": 0,
                   "none": 0
                   }
    # Convert input to lowercase and split if multiple diseases provided (e.g., "diabetes & thyroid")
    diseases = medical_history.lower().split('&')

    # Sum up the risk scores for all diseases in the input
    total_risk_score = sum(risk_scores.get(disease,0) for disease in diseases)
    # Define normalization range (min-max scaling)
    max_score = 14
    min_score = 0

    normalized_risk_score  = (total_risk_score-min_score)/(max_score-min_score)

    return  normalized_risk_score

# Function: preprocessing_input
# - Converts raw user inputs into a feature-engineered DataFrame
#   that matches the model's expected format.
# - Handles one-hot encoding for categorical variables.
# - Encodes insurance plan and calculates normalized risk.

def preprocessing_input(input_dict):
    # Expected feature columns for the model
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]
    # Label-Encoding Insurance Plan
    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
    # Initialize DataFrame with all zeros
    df = pd.DataFrame(0,columns=expected_columns, index=[0])
    # Map each input field to the correct encoded column
    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        elif key == 'BMI Category':
            if value == 'Obesity':
                df['bmi_category_Obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_Underweight'] = 1
        elif key == 'Smoking Status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1
        elif key == 'Insurance Plan':  # Correct key usage with case sensitivity
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':  # Correct key usage with case sensitivity
            df['age'] = value
        elif key == 'Number of Dependants':  # Correct key usage with case sensitivity
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':  # Correct key usage with case sensitivity
            df['income_lakhs'] = value
        elif key == "Genetical Risk":
            df['genetical_risk'] = value
    # Add normalized medical history risk
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])
    # Apply scaling (young vs rest) to numerical columns
    df = handle_scaling(input_dict['Age'],df)

    return df

# Function: handle_scaling
# - Selects the correct scaler based on age group.
# - Scales only specific columns to standardize input features.

def handle_scaling(age,df):
    # Pick the appropriate scaler
    if age<=25:
        scaler_object = scaler_young

    else:
        scaler_object = scaler_rest
    # Get columns to scale and the actual scaler
    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']
    # Placeholder for scaling (removed later)
    df['income_level'] = None
    # Apply scaler transformation
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    # Drop placeholder column

    df.drop('income_level', axis=1,inplace=True)

    return df
# Function: predict
# - Main function that calls preprocessing and selects the
#   correct model (young vs rest) to generate prediction.
def predict(input_dict):
    # Preprocess the input dictionary into a model-ready DataFrame
    input_df = preprocessing_input(input_dict)
    # Choose model based on age group
    if input_dict['Age'] <=25 :
        prediction = model_young.predict(input_df)

    else:
        prediction = model_rest.predict(input_df)
    # Return integer value of predicted insurance cost
    return int(prediction[0])





