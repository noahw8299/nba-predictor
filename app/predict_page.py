import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import os
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import stat_viz as stats

# Cache the model and scaler loading functions
@st.cache_resource
def load_model():
    try:
        model_path = get_dir_path('predict_model.pkl')
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

@st.cache_resource
def load_scaler():
    try:
        scaler_path = get_dir_path('model_scaler.pkl')
        scaler = joblib.load(scaler_path)
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
    
    #print(recent_stats.columns)
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

    feature_vector_scaled_df = pd.DataFrame(feature_vector_scaled, columns=numerical_columns)
    
    return feature_vector_scaled_df

def get_dir_path(file):
    # Get the current working directory
    cwd = os.getcwd()

    # Create the full path to the CSV file
    return os.path.join(cwd, "app", file) 

def show_predict_page():
    st.title("Predict NBA Games")

    st.write("""### Choose two teams""")
    teams = {
        "POR", "PHI", "PHO", "SAC", "DEN", "ORL", "GSW", "DET", "MIL", "CHI", "WAS", "MIA",
        "LAL", "NOP", "ATL", "BRK", "MIN", "UTA", "IND", "SAS", "CLE", "NYK", "TOR", "CHO",
        "HOU", "DAL", "BOS", "MEM", "LAC", "OKC"
    }

    # Create the full path to the CSV file
    csv_path = get_dir_path("result.csv")

    # Read the CSV file
    data = pd.read_csv(csv_path)

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
            home_stats = get_recent_performance_stats(home_team, 'home', features)
            away_stats = get_recent_performance_stats(away_team, 'away', features)
            feature_vector = prepare_features_for_prediction(home_team, away_team, features, scaler, numerical_columns)
            if feature_vector is not None:
                predicted_spread = model.predict(feature_vector)
                st.write(f"Predicted spread: {predicted_spread[0]}")

                # Show additional visualizations
                stats.create_dashboard(home_stats, away_stats)

                
        else:
            st.error('Please enter both home and away teams.')