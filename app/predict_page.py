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
    # Create the full path to the CSV file
    csv_path = get_dir_path("result.csv")

    # Read the CSV file
    data = pd.read_csv(csv_path, parse_dates=['date'])

    # Define the features - include team identifiers
    # Define stats that will be unavailable when trying to make predictions on the future
    unknown = ['total', 'total_opp', 'won', 'home_pts_scored', 'away_pts_scored', 'home_pts_allowed', 'away_pts_allowed']

    features = data.drop(columns=['spread', 'date', 'season'] + unknown)
    # Identify numerical columns for scaling
    numerical_columns = features.select_dtypes(include=['float64', 'int64']).columns

    model = load_model()
    scaler = load_scaler()

    # Extracting the most recent game details
    most_recent_game = data.iloc[-1]

    # Getting the year from the most recent game's date
    end_year = most_recent_game['date'].year

    # Calculating the score details
    home_team = most_recent_game['home_team']
    away_team = most_recent_game['away_team']
    home_pts_scored = most_recent_game['home_pts_scored']
    away_pts_scored = most_recent_game['away_pts_scored']
    date = most_recent_game['date'].strftime('%Y-%m-%d')

    if home_pts_scored > away_pts_scored:
        winner = home_team
        loser = away_team
        spread = home_pts_scored - away_pts_scored
    else:
        winner = away_team
        loser = home_team
        spread = away_pts_scored - home_pts_scored

    # Constructing the intro message with Markdown
    st.markdown(f"""
    # NBA Game Prediction Model

    ### Introduction
    This model is a **random forest model** trained on NBA game box scores from the **2014 to {end_year} season**. 
    """)

    # Adding a horizontal line
    st.markdown("---")

    # Creating columns for better use of space
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"### Most Recent Game")
        st.markdown(f"**Date:** {date}")
        st.markdown(f"**Teams:** {home_team} vs. {away_team}")

    with col2:
        st.markdown(f"### Score")
        st.markdown(f"**Score:** {home_team} {home_pts_scored} - {away_team} {away_pts_scored}")

    with col3:
        st.markdown(f"### Outcome")
        st.markdown(f"**Winner:** :trophy: **{winner}** :trophy: ")
        st.markdown(f"**Spread:** {winner} won by **{spread} points**")

    # Adding some spacing
    st.markdown("---")

    # Optionally adding a container for additional info
    with st.container():
        st.markdown("### Model Details")
        st.write("The model uses a random forest algorithm to predict game outcomes based on historical box score data. It has been trained on data from the 2014 to 2023 NBA seasons, capturing various player and team statistics to make accurate predictions.")

    # Optional: adding an expander for more details
    with st.expander("See more details"):
        st.write("""
            - **Algorithm:** Random Forest
            - **Training Data:** NBA game box scores (2014-2023)
            - **Features:** Player and team statistics
        """)

    if model is None or scaler is None:
        st.error("Model or scaler could not be loaded. Please check the files.")
        return

    st.title("Predict NBA Games")

    st.write("""### Choose two teams""")
    teams = {
        "POR", "PHI", "PHO", "SAC", "DEN", "ORL", "GSW", "DET", "MIL", "CHI", "WAS", "MIA",
        "LAL", "NOP", "ATL", "BRK", "MIN", "UTA", "IND", "SAS", "CLE", "NYK", "TOR", "CHO",
        "HOU", "DAL", "BOS", "MEM", "LAC", "OKC"
    }

    col1, col2 = st.columns(2)

    with col1:
        home_team = st.selectbox("Home", teams)

    with col2:
        away_team = st.selectbox("Away", teams)

    ok = st.button("PREDICT")

    # Adding some spacing
    st.markdown("---")

    if ok:
        if home_team and away_team:
            home_stats = get_recent_performance_stats(home_team, 'home', features)
            away_stats = get_recent_performance_stats(away_team, 'away', features)
            feature_vector = prepare_features_for_prediction(home_team, away_team, features, scaler, numerical_columns)
            if feature_vector is not None:
                predicted_spread = model.predict(feature_vector)
                st.write(f"""### Predicted Spread {predicted_spread[0]}""")
                        
                # Show additional visualizations
                selected_metrics = ['fgpct', '3ppct', 'ftpct']
                labels = ['Field Goal %', '3-Point %', 'Free Throw %'] + [home_team, away_team]
                stats.create_dashboard(home_stats, away_stats, selected_metrics, labels)

                st.write(away_stats)

                cols = ['rolling_avg_fg_opp','rolling_avg_fga_opp','rolling_avg_3p_opp','rolling_avg_3pa_opp','rolling_avg_ft_opp','rolling_avg_fta_opp']
                stats.create_box_chart(home_stats, away_stats, cols)
                
        else:
            st.error('Please enter both home and away teams.')