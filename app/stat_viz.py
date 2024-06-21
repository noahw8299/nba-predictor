import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from math import pi

def create_dashboard(home_stats, away_stats, selected_metrics, labels, home_team='home', away_team='away'):
    # Select key metrics for comparison

    home_values = home_stats[selected_metrics].values.flatten().tolist()
    away_values = away_stats[[f"{stat}_opp" for stat in selected_metrics]].values.flatten().tolist()

    fig = go.Figure()

    # Add traces with customized line colors
    fig.add_trace(go.Scatterpolar(
        r=home_values,
        theta=labels,
        fill='toself',
        name=f'Home ({home_team})',
        line=dict(color='blue')  # Set color to blue for home team
    ))
    fig.add_trace(go.Scatterpolar(
        r=away_values,
        theta=labels,
        fill='toself',
        name=f'Away ({away_team})',
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

def create_bar_chart(home_stats, away_stats, cols, home_team="Team 1", away_team="Team 2"):
    fig = go.Figure()

    # Convert home_stats and away_stats to DataFrames if they are not already
    if not isinstance(home_stats, pd.DataFrame):
        home_stats = home_stats.to_frame().T
    if not isinstance(away_stats, pd.DataFrame):
        away_stats = away_stats.to_frame().T

    # Filter columns for home and away teams based on the provided substrings
    home_cols = [col for col in home_stats.columns if any(sub in col for sub in cols)]
    away_cols = [col for col in away_stats.columns if any(sub in col for sub in cols)]

    # Remove '_opp' suffix for proper grouping
    home_cols_clean = [col.replace("home_rolling_avg_", "") for col in home_cols]
    away_cols_clean = [col.replace("away_rolling_avg_", "").replace("_opp", "") for col in away_cols]

    categories = list(set(home_cols_clean + away_cols_clean))
    categories.sort()

    home_values = []
    away_values = []
    for category in categories:
        home_col = f'home_rolling_avg_{category}'
        away_col = f'away_rolling_avg_{category}_opp'

        if home_col in home_stats:
            home_values.append(home_stats[home_col].values[0])
        else:
            home_values.append(0)

        if away_col in away_stats:
            away_values.append(away_stats[away_col].values[0])
        else:
            away_values.append(0)

    fig.add_trace(go.Bar(
        x=categories,
        y=home_values,
        name=f'{home_team}',
        marker_color='blue'  # Set color to blue for home team
    ))

    fig.add_trace(go.Bar(
        x=categories,
        y=away_values,
        name=f'{away_team}',
        marker_color='red'  # Set color to red for away team
    ))

    fig.update_layout(
        title="Comparison of Rolling Averages Over the Past 5 Games",
        yaxis_title="Value",
        xaxis_title="Statistics",
        barmode='group',  # Group bars of the same category together
        showlegend=True
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