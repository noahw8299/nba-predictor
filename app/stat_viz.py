import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from math import pi

def create_dashboard(home_stats, away_stats, selected_metrics, labels):
    # Select key metrics for comparison

    home_values = home_stats[selected_metrics].values.flatten().tolist()
    away_values = away_stats[[f"{stat}_opp" for stat in selected_metrics]].values.flatten().tolist()

    fig = go.Figure()

    teams = labels[3:]
    labels = labels[:-2]

    # Add traces with customized line colors
    fig.add_trace(go.Scatterpolar(
        r=home_values,
        theta=labels,
        fill='toself',
        name=f'Home ({teams[0]})',
        line=dict(color='blue')  # Set color to blue for home team
    ))
    fig.add_trace(go.Scatterpolar(
        r=away_values,
        theta=labels,
        fill='toself',
        name=f'Away ({teams[1]})',
        line=dict(color='red')  # Set color to red for away team
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Shooting Percentages Comparison"
    )

    st.plotly_chart(fig)

def create_box_chart(home_stats, away_stats, cols, team1_name="Team 1", team2_name="Team 2"):
    fig = go.Figure()

    # Convert home_stats and away_stats to DataFrames if they are not already
    if not isinstance(home_stats, pd.DataFrame):
        home_stats = home_stats.to_frame()
    if not isinstance(away_stats, pd.DataFrame):
        away_stats = away_stats.to_frame()

    # Filter columns for home and away teams based on the provided substrings
    home_cols = [col for col in home_stats.columns if any(sub in col for sub in cols)]
    away_cols = [col for col in away_stats.columns if any(sub in col for sub in cols)]

    # Add box plot traces for each home column and team
    for col in home_cols:
        fig.add_trace(go.Box(
            y=home_stats[col],
            name=f'{team1_name} - {col.replace("home_rolling_avg_", "")}',
            boxmean='sd',  # Shows the mean and standard deviation
            marker_color='blue'  # Set color to blue for team 1
        ))

    # Add box plot traces for each away column and team
    for col in away_cols:
        fig.add_trace(go.Box(
            y=away_stats[col],
            name=f'{team2_name} - {col.replace("away_rolling_avg_", "")}',
            boxmean='sd',  # Shows the mean and standard deviation
            marker_color='red'  # Set color to red for team 2
        ))

    fig.update_layout(
        title="Comparison of Rolling Averages",
        yaxis_title="Value",
        xaxis_title="Statistics",
        boxmode='group',  # Group boxes of the same category together
        showlegend=False
    )

    st.plotly_chart(fig)

def plot_head_to_head(data, home_team, away_team):
    home_data = data[data['home_team'] == home_team].select_dtypes(include=['float64', 'int64']).mean()
    away_data = data[data['away_team'] == away_team].select_dtypes(include=['float64', 'int64']).mean()

    metrics = ['pts_scored', 'pts_allowed', 'rebounds', 'assists']
    home_metrics = home_data[[f'home_{metric}' for metric in metrics]]
    away_metrics = away_data[[f'away_{metric}' for metric in metrics]]

    metrics_df = pd.DataFrame({
        'Metrics': metrics,
        'Home': home_metrics.values,
        'Away': away_metrics.values
    })

    metrics_df.plot(kind='bar', x='Metrics', figsize=(10, 5))
    plt.title('Head-to-Head Comparison')
    plt.ylabel('Average per Game')
    st.pyplot(plt)

def show_win_probability(predicted_spread):
    win_prob_home = 1 / (1 + np.exp(-predicted_spread)) * 100  # Assuming logistic regression output
    win_prob_away = 100 - win_prob_home

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = win_prob_home,
        title = {'text': "Home Team Win Probability"},
        gauge = {'axis': {'range': [0, 100]}}
    ))

    st.plotly_chart(fig)