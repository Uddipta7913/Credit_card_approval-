import streamlit as st
import pandas as pd
import numpy as np
import hashlib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("indian_credit_cards_30.csv")

df = load_data()

# --- Train Model ---
@st.cache_resource
def train_model(data):
    # Define features + target
    categorical_cols = ['Bank', 'Card_Name', 'Card_Type', 'Rewards', 'Audience']
    numeric_cols = ['Annual_Fee', 'card_id', 'credit_score_min', 'income_min']
    target_col = 'Difficulty'   # make sure your dataset has this column

    X = data[categorical_cols + numeric_cols]
    y = data[target_col]

    # Transformer: OneHotEncode categoricals, passthrough numerics
    ct = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numeric_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_transformed = ct.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train_transformed, y_train)

    return ct, model

ct, model = train_model(df)

# --- Prediction Function ---
def predict_difficulty(user_data, column_transformer, model):
    try:
        user_df = pd.DataFrame([user_data])
        user_processed = column_transformer.transform(user_df)
        prediction = model.predict(user_processed)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return "Error"

# --- Recommendation Function ---
def recommend_cards(predicted_difficulty, card_df):
    if predicted_difficulty == "Error":
        return pd.DataFrame()
    return card_df[card_df['Difficulty'] == predicted_difficulty].copy()

# --- Utility: Color Generator ---
def string_to_color(s):
    if pd.isna(s):
        s = "Unknown"
    hex_dig = hashlib.md5(str(s).encode()).hexdigest()
    return f"#{hex_dig[:6]}"

# --- Streamlit UI ---
st.set_page_config(page_title="Credit Card Recommender", layout="wide")

st.title("💳 Credit Card Eligibility and Recommendation")
st.write("Find the perfect credit card for you based on your profile.")

st.sidebar.header("Your Information")

# Sidebar options from dataset
issuer_options = df['Bank'].dropna().unique().tolist()
card_type_options = df['Card_Type'].dropna().unique().tolist()
audience_options = df['Audience'].dropna().unique().tolist()

issuer = st.sidebar.selectbox("Select Card Issuer (Bank)", issuer_options)
card_type = st.sidebar.selectbox("Select Card Type", card_type_options)
audience = st.sidebar.selectbox("Your Audience Type", audience_options)

annual_fee = st.sidebar.number_input("Enter Annual Fee (₹)", min_value=0, value=0)
credit_score_min_input = st.sidebar.number_input("Enter Your Credit Score", min_value=0, max_value=900, value=700)
income_min_input = st.sidebar.number_input("Enter Your Annual Income (₹)", min_value=0, value=300000)

# Build user input
user_input_data = {
    'Bank': issuer,
    'Card_Name': None,  # placeholder
    'Card_Type': card_type,
    'Rewards': None,    # placeholder
    'Audience': audience,
    'Annual_Fee': annual_fee,
    'card_id': 0,  # placeholder
    'credit_score_min': credit_score_min_input,
    'income_min': income_min_input
}

# Run prediction
if st.sidebar.button("Find My Recommended Cards"):
    predicted_difficulty = predict_difficulty(user_input_data, ct, model)
    st.subheader(f"Predicted Approval Difficulty: **{predicted_difficulty}**")

    recommended_cards = recommend_cards(predicted_difficulty, df)

    if not recommended_cards.empty:
        st.subheader("Recommended Cards for You:")
        for _, row in recommended_cards.iterrows():
            card_color = string_to_color(row['Card_Name'])
            st.markdown(f"""
            <div style="background-color:{card_color}; padding:15px; border-radius:10px; margin-bottom:10px;">
                <b>{row['Card_Name']}</b><br>
                Issuer: {row['Bank']}<br>
                Type: {row['Card_Type']}<br>
                Annual Fee: ₹{row['Annual_Fee']}<br>
                Min Credit Score: {row['credit_score_min']}<br>
                Min Income: ₹{row['income_min']}<br>
                Rewards: {row['Rewards']}<br>
                Features: {row['features']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No matching cards found for your profile.")

