import streamlit as st
import pandas as pd
import numpy as np
import pickle # Import pickle to load objects
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# Assuming you used RandomForestClassifier, import it
from sklearn.ensemble import RandomForestClassifier
import hashlib # Import hashlib to create a consistent color based on card name

# --- Train ColumnTransformer & RandomForest on the dataset ---
from sklearn.model_selection import train_test_split


@st.cache_resource  # cache so it doesn’t retrain every time
def train_model(data):
    # Define features and target
    categorical_cols = ['Bank','Card_Name','Card_Type','Rewards','Audience','Annual_Fee']
    numeric_cols = ['card_id', 'credit_score_min', 'income_min']
    target_col = 'Difficulty'   # assuming you labeled this during training

    X = data[categorical_cols + numeric_cols]
    y = data[target_col]

    # Preprocess: OneHotEncode categorical + pass numeric
    ct = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numeric_cols)
        ]
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit transformer + model
    X_train_transformed = ct.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_transformed, y_train)

    return ct, model

# Load data and train
df = pd.read_csv("indian_credit_cards_30.csv")
ct, model = train_model(df)



# --- Load Data and Model ---
try:
    # Load the original data to get unique values for selectboxes
    df = pd.read_csv("indian_credit_cards_30.csv")

    # Load your saved ColumnTransformer and model using pickle
    with open('column_transformer.pkl', 'rb') as f:
        ct = pickle.load(f)
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Removed the success message here as requested
    # st.success("Data, ColumnTransformer, and Model loaded successfully.")

except FileNotFoundError:
    st.error("Error: Model or ColumnTransformer files not found. Please run the training cell in your Colab notebook first to save them.")
    st.stop() # Stop execution if files are not found
except Exception as e:
    st.error(f"Error loading data or model: {e}")
    st.stop() # Stop execution if data or model loading fails

# --- Prediction and Recommendation Functions ---
def predict_difficulty(user_data, column_transformer, model):
    """
    Predicts the credit card approval difficulty for a given user.
    Ensures user input matches the structure expected by the transformer.
    """
    if column_transformer is None or model is None:
        st.error("Model or ColumnTransformer not loaded.")
        return "Error"

    # Define the exact list of columns that the ColumnTransformer was fitted on during training.
    # Get this list from the transformer's fitted attributes if possible,
    # but based on the training code, it was fitted on:
    # categorical_cols = ['Bank','Card_Name','Card_Type','Rewards','Audience','Annual_Fee']
    # numeric_cols = ['card_id', 'credit_score_min', 'income_min']
    # The order in which these columns were passed to the transformer matters.
    # Let's assume the order was categorical_cols followed by numeric_cols.
    # We need to recreate a DataFrame with these columns in the same order.

    # Columns the transformer was fitted on, in the expected order
    transformer_fitted_cols = ['Bank', 'Card_Name', 'Card_Type', 'Rewards', 'Audience', 'Annual_Fee', 'card_id', 'credit_score_min', 'income_min']

    # Create a dictionary to hold the data for the new DataFrame
    # Initialize with None or appropriate default values
    user_data_processed_dict = {col: None for col in transformer_fitted_cols}

    # Populate the dictionary with user provided data
    # Ensure the keys match the column names expected by the transformer
    user_data_processed_dict['Bank'] = user_data.get('Bank', None)
    user_data_processed_dict['Card_Type'] = user_data.get('Card_Type', None)
    user_data_processed_dict['Audience'] = user_data.get('Audience', None)
    user_data_processed_dict['Annual_Fee'] = user_data.get('Annual_Fee', 0) # Assuming 0 as a default for numeric fee
    user_data_processed_dict['credit_score_min'] = user_data.get('credit_score_min', 0) # Assuming 0 as a default for numeric score
    user_data_processed_dict['income_min'] = user_data.get('income_min', 0) # Assuming 0 as a default for numeric income

    # For columns not provided by the user ('Card_Name', 'Rewards', 'card_id'),
    # they will remain None or 0 as initialized, which should be handled by handle_unknown='ignore'
    # for categorical and treated as numeric for 'card_id'.

    # Create the DataFrame with the correct columns and data
    user_df_processed = pd.DataFrame([user_data_processed_dict], columns=transformer_fitted_cols)

    # Ensure data types are consistent, especially for categorical columns
    for col in ['Bank', 'Card_Name', 'Card_Type', 'Rewards', 'Audience']:
         if col in user_df_processed.columns:
             user_df_processed[col] = user_df_processed[col].astype(object)

    try:
        # Transform the user data using the loaded ColumnTransformer
        user_processed = column_transformer.transform(user_df_processed)
        # Make prediction using the loaded model
        prediction = model.predict(user_processed)
        return prediction[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.error("Please ensure your input data structure matches the model's training data.")
        st.error(f"DataFrame structure being transformed: {user_df_processed.head()}") # Add this for debugging
        return "Error"


def recommend_cards(predicted_difficulty, card_df):
    """
    Recommends credit cards based on the predicted approval difficulty.
    """
    if predicted_difficulty == "Error":
        return pd.DataFrame() # Return empty dataframe if prediction failed

    recommended_cards = card_df[card_df['Difficulty'] == predicted_difficulty].copy()
    return recommended_cards

# Function to generate a color based on a string
def string_to_color(s):
    """Generates a consistent hex color based on a string."""
    # Use hashlib to get a consistent hash
    if pd.isna(s): # Handle potential NaN values
        s = "Unknown"
    hash_object = hashlib.md5(str(s).encode()) # Ensure s is a string for encoding
    hex_dig = hash_object.hexdigest()
    # Take the first 6 characters to form a hex color code
    return f"#{hex_dig[:6]}"


# --- Streamlit UI ---
st.set_page_config(page_title="Credit Card Recommender", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px; /* Reduced padding */
        margin-bottom: 10px; /* Reduced margin */
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1); /* Slightly smaller shadow */
        background-color: #fff;
        transition: transform 0.3s ease-in-out;
        width: 300px; /* Set a fixed width for a smaller box */
        display: inline-block; /* Display cards side-by-side */
        vertical-align: top; /* Align cards at the top */
        margin-right: 15px; /* Add space between cards */
    }
     .card:hover {
        transform: translateY(-5px);
    }
    .card-title {
        font-size: 16px; /* Reduced font size */
        font-weight: bold;
        color: #333;
        margin-bottom: 3px; /* Reduced margin */
    }
    .card-issuer {
        font-size: 14px; /* Increased font size */
        color: #333; /* Darker color */
        margin-bottom: 2px; /* Reduced margin */
    }
    .card-type {
        font-size: 14px; /* Increased font size */
        color: #333; /* Darker color */
        margin-bottom: 2px; /* Reduced margin */
    }
    .card-fee {
        font-size: 14px; /* Increased font size */
        color: #c0392b;
        margin-bottom: 2px; /* Reduced margin */
    }
    .card-score {
        font-size: 14px; /* Increased font size */
        color: #2980b9;
        margin-bottom: 2px; /* Reduced margin */
    }
    .card-income {
        font-size: 14px; /* Increased font size */
        color: #27ae60;
        margin-bottom: 8px; /* Reduced margin */
    }
    .card p {
        font-size: 14px; /* Increased font size */
        color: #333; /* Darker color */
        margin-bottom: 3px; /* Reduced margin */
    }
    </style>
""", unsafe_allow_html=True)

st.title("💳 Credit Card Eligibility and Recommendation")
st.write("Find the perfect credit card for you based on your profile.")

st.sidebar.header("Your Information")

# --- User Inputs (Based on your features used by the model) ---
# Use unique values from the loaded dataframe for selectboxes
issuer_options = df['Bank'].unique().tolist() if df is not None and 'Bank' in df.columns else []
card_type_options = df['Card_Type'].unique().tolist() if df is not None and 'Card_Type' in df.columns else []
audience_options = df['Audience'].unique().tolist() if df is not None and 'Audience' in df.columns else []

# Add a default option for selectboxes if the list is empty or for better UX
issuer_options.insert(0, 'Select...') if 'Select...' not in issuer_options else None
card_type_options.insert(0, 'Select...') if 'Select...' not in card_type_options else None
audience_options.insert(0, 'Select...') if 'Select...' not in audience_options else None


issuer = st.sidebar.selectbox("Select Card Issuer (Bank)", issuer_options)
card_type = st.sidebar.selectbox("Select Card Type", card_type_options)
audience = st.sidebar.selectbox("Your Audience Type", audience_options)

# Get numeric inputs - ensure they match the column names used by the model
annual_fee = st.sidebar.number_input("Enter Annual Fee (if any)", min_value=0, value=0)
credit_score_min_input = st.sidebar.number_input("Enter Your Credit Score", min_value=0, max_value=900, value=700)
income_min_input = st.sidebar.number_input("Enter Your Annual Income (approx)", min_value=0, value=300000)

# Note: 'card_name', 'reward_rate', and 'card_id' were also used by the transformer in training.
# These are NOT inputs from the user. We need to provide placeholder values for these columns
# in the user_data dictionary that the trained transformer can handle during inference.
# Since handle_unknown='ignore' is used for OneHotEncoder, None should be a safe placeholder
# for the categorical columns ('Card_Name', 'Rewards').
# For 'card_id' (numeric), 0 is used as a placeholder.

# User data dictionary - ensure keys match the exact column names expected by the transformer
# This dictionary is used to pass user inputs to the predict_difficulty function
user_input_data = {
    'Bank': issuer if issuer != 'Select...' else None, # Map sidebar input to correct column name
    'Card_Type': card_type if card_type != 'Select...' else None, # Map sidebar input
    'Audience': audience if audience != 'Select...' else None, # Map sidebar input
    'Annual_Fee': annual_fee, # Map sidebar input
    'credit_score_min': credit_score_min_input, # Map sidebar input
    'income_min': income_min_input, # Map sidebar input
}


st.sidebar.markdown("---")
if st.sidebar.button("Find My Recommended Cards"):
    # Validate that required selectboxes are selected
    if issuer == 'Select...' or card_type == 'Select...' or audience == 'Select...':
        st.sidebar.warning("Please select values for Issuer, Card Type, and Audience.")
    elif ct is not None and model is not None:
        # Use the user_input_data dictionary
        predicted_difficulty = predict_difficulty(user_input_data, ct, model)

        st.subheader(f"Predicted Approval Difficulty: **{predicted_difficulty}**")

        if predicted_difficulty != "Error":
            recommended_cards = recommend_cards(predicted_difficulty, df)

            if not recommended_cards.empty:
                st.subheader("Recommended Cards for You:")
                # Display recommended cards in a nice format with dynamic colors
                for index, row in recommended_cards.iterrows():
                    card_color = string_to_color(row['Card_Name']) # Generate color based on card name
                    st.markdown(f"""
                    <div class="card" style="background-color: {card_color};">
                        <div class="card-title">{row['Card_Name']}</div>
                        <div class="card-issuer">Issuer: {row['Bank']}</div>
                        <div class="card-type">Type: {row['Card_Type']}</div>
                        <div class="card-fee">Annual Fee: ₹{row['Annual_Fee']}</div>
                        <div class="card-score">Min Credit Score: {row['credit_score_min']}</div>
                        <div class="card-income">Min Income: ₹{row['income_min']}</div>
                        <p>Features: {row['features']}</p>
                        <p>Reward Rate: {row['Rewards']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"No cards found with '{predicted_difficulty}' difficulty in the dataset.")
    else:
        st.error("Model or ColumnTransformer not loaded. Cannot make prediction.")
