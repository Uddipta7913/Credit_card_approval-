import streamlit as st
import pandas as pd
import numpy as np
import pickle # Import pickle to load objects
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# Assuming you used RandomForestClassifier, import it
from sklearn.ensemble import RandomForestClassifier
import hashlib # Import hashlib to create a consistent color based on card name


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

    user_df = pd.DataFrame([user_data])

    # Define the exact list of columns that the ColumnTransformer was fitted on.
    # This must match the order and names used in the training notebook.
    # Based on the training code, the columns used for fitting ct are:
    # categorical_cols = ['Bank','Card_Name','Card_Type','Rewards','Audience','Annual_Fee']
    # numeric_cols = ['card_id', 'credit_score_min', 'income_min']
    # So, the columns for transformation are:
    expected_cols_for_transform = ['Bank', 'Card_Name', 'Card_Type', 'Rewards', 'Audience', 'Annual_Fee', 'card_id', 'credit_score_min', 'income_min']

    # Reindex the user dataframe to match the expected columns.
    # Fill missing columns that are not user input (like card_id, Card_Name, Rewards)
    # with appropriate placeholder values that the transformer can handle (e.g., None, or a value that OneHotEncoder will ignore).
    # Using None for categorical columns when handle_unknown='ignore' is set in OneHotEncoder is often okay.
    user_df_processed = user_df.reindex(columns=expected_cols_for_transform) # .reindex will add missing columns and fill with NaN by default

    # You might need to fill NaNs for numeric columns if they are not provided by the user
    # For simplicity here, let's assume if a user input is missing it's NaN and transformer handles it.
    # If your transformer expects specific fill values (like 0 for income_min if missing), add that here:
    # user_df_processed['income_min'] = user_df_processed['income_min'].fillna(0) # Example

    # Convert relevant columns to object type to avoid potential type issues with None/NaN during transformation
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
    hash_object = hashlib.md5(s.encode())
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

# Note: 'card_name', 'reward_rate', and 'card_id' were also used by the transformer.
# Asking the user for all card names or reward rates is not practical for a general recommender.
# We need to provide placeholder values for these columns in the user_data dictionary
# that the trained transformer can handle.
# Since handle_unknown='ignore' is used, setting them to None might work for categorical.
# For 'card_id' (numeric), 0 or a similar placeholder might be needed depending on training.
# Let's create placeholder values that should be compatible with the transformer fitted on the training data.
# These values won't affect the prediction significantly if they are outside of the trained range/categories
# and the transformer is configured to ignore unknown values.

# User data dictionary - ensure keys match the exact column names expected by the transformer
user_data = {
    'Bank': issuer if issuer != 'Select...' else None, # Map sidebar input to correct column name
    'Card_Type': card_type if card_type != 'Select...' else None, # Map sidebar input
    'Audience': audience if audience != 'Select...' else None, # Map sidebar input
    'Annual_Fee': annual_fee, # Map sidebar input
    'credit_score_min': credit_score_min_input, # Map sidebar input
    'income_min': income_min_input, # Map sidebar input
    # Placeholder values for columns used by the transformer but not taken as direct user input
    'Card_Name': 'Placeholder', # Use a placeholder name
    'Rewards': 'Placeholder', # Use a placeholder reward rate
    'card_id': 0 # Use a placeholder ID (assuming 0 or similar was not a significant card_id in training)
}


st.sidebar.markdown("---")
if st.sidebar.button("Find My Recommended Cards"):
    # Validate that required selectboxes are selected
    if issuer == 'Select...' or card_type == 'Select...' or audience == 'Select...':
        st.sidebar.warning("Please select values for Issuer, Card Type, and Audience.")
    elif ct is not None and model is not None:
        # Use the corrected user_data dictionary
        predicted_difficulty = predict_difficulty(user_data, ct, model)

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
