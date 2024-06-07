import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn
import os
import joblib

# Cache the model and scaler loading functions
@st.cache_resource
def load_model():
    try:
        model = joblib.load('predict_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load('model_scaler.pkl')
        return scaler
    except Exception as e:
        st.error(f"Error loading the scaler: {e}")
        return None

def get_recent_performance_stats(team, team_type, data):
    if team_type == 'home':
        team_data = data[data['home_team'] == team]
        team_data = team_data.select_dtypes(include=['float64', 'int64'])
        recent_stats = team_data.iloc[-1]
        recent_stats = recent_stats[[col for col in recent_stats.index if 'opp' not in col and 'away' not in col]]
    elif team_type == 'away':
        team_data = data[data['away_team'] == team]
        team_data = team_data.select_dtypes(include=['float64', 'int64'])
        recent_stats = team_data.iloc[-1]
        recent_stats = recent_stats[[col for col in recent_stats.index if 'opp' in col or 'away' in col]]
    else:
        raise ValueError("team_type must be 'home' or 'away'")
    
    if team_data.empty:
        raise ValueError(f"No recent performance stats found for team {team} as {team_type}.")
    
    return recent_stats

def prepare_features_for_prediction(home_team, away_team, data, scaler, numerical_columns):
    try:
        home_stats = get_recent_performance_stats(home_team, 'home', data)
        away_stats = get_recent_performance_stats(away_team, 'away', data)
    except ValueError as e:
        st.error(f"Error in getting recent performance stats: {e}")
        return None
    
    # Concatenate the stats
    feature_vector = pd.concat([home_stats, away_stats]).to_frame().T
    feature_vector = feature_vector[numerical_columns]
    
    # Scale the feature vector
    feature_vector_scaled = scaler.transform(feature_vector)
    
    return feature_vector_scaled

def show_predict_page():
    st.title("Predict NBA Games")

    st.write("""### Choose two teams""")
    teams = {
        "POR", "PHI", "PHO", "SAC", "DEN", "ORL", "GSW", "DET", "MIL", "CHI", "WAS", "MIA",
        "LAL", "NOP", "ATL", "BRK", "MIN", "UTA", "IND", "SAS", "CLE", "NYK", "TOR", "CHO",
        "HOU", "DAL", "BOS", "MEM", "LAC", "OKC"
    }

    data = pd.read_csv("result.csv")

    # Define the features - include team identifiers
    # Define stats that will be unavailable when trying to make predictions on the future
    unknown = ['total', 'total_opp', 'won', 'home_pts_scored', 'away_pts_scored', 'home_pts_allowed', 'away_pts_allowed']

    features = data.drop(columns=['spread', 'date', 'season'] + unknown)
    # Identify numerical columns for scaling
    numerical_columns = features.select_dtypes(include=['float64', 'int64']).columns

    model = load_model()
    scaler = load_scaler()

    if model is None or scaler is None:
        st.error("Model or scaler could not be loaded. Please check the files.")
        return

    col1, col2 = st.columns(2)

    with col1:
        home_team = st.selectbox("Home", teams)

    with col2:
        away_team = st.selectbox("Away", teams)

    ok = st.button("PREDICT")

    if ok:
        if home_team and away_team:
            feature_vector = prepare_features_for_prediction(home_team, away_team, data, scaler, numerical_columns)
            if feature_vector is not None:
                predicted_spread = model.predict(feature_vector)
                st.write(f"Predicted spread: {predicted_spread[0]}")
        else:
            st.error('Please enter both home and away teams.')