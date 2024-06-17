import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

def plot_shooting_percentages(home_stats, away_stats):
    shooting_stats = ['fgpct', '3ppct', 'ftpct']
    home_shooting = home_stats[shooting_stats]
    away_shooting = away_stats[[f"{stat}_opp" for stat in shooting_stats]]

    fig, ax = plt.subplots()
    width = 0.35  # the width of the bars

    # Plot home shooting percentages
    home_bars = ax.barh(shooting_stats, home_shooting, width, label='Home', color='blue')

    # Plot away shooting percentages
    away_bars = ax.barh([f"{stat}_opp" for stat in shooting_stats], away_shooting, width, label='Away', color='orange')

    ax.set_xlabel('Percentage')
    ax.set_title('Shooting Percentages')
    ax.legend()

    st.pyplot(fig)

def plot_radar_chart(home_stats, away_stats):
    from math import pi
    
    labels = ['fgpct', '3ppct', 'ftpct']
    home_values = home_stats[labels].values.flatten().tolist()
    away_values = away_stats[[f"{stat}_opp" for stat in labels]].values.flatten().tolist()

    # Number of variables
    num_vars = len(labels)

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    # Initialize the radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], labels)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
    plt.ylim(0, 1)

    # Plot data
    home_values += home_values[:1]
    away_values += away_values[:1]

    ax.plot(angles, home_values, linewidth=1, linestyle='solid', label="Home")
    ax.fill(angles, home_values, 'b', alpha=0.1)

    ax.plot(angles, away_values, linewidth=1, linestyle='solid', label="Away")
    ax.fill(angles, away_values, 'r', alpha=0.1)

    plt.title('Shooting Percentages Comparison')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    st.pyplot(fig)

def create_dashboard(home_stats, away_stats):
    # Select key metrics for comparison
    selected_metrics = ['fgpct', '3ppct', 'ftpct']
    home_values = home_stats[selected_metrics].values.flatten().tolist()
    away_values = away_stats[[f"{stat}_opp" for stat in selected_metrics]].values.flatten().tolist()
    
    # Ensure the metrics and values have consistent names for readability
    labels = ['Field Goal %', '3-Point %', 'Free Throw %']

    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatterpolar(
        r=home_values,
        theta=labels,
        fill='toself',
        name='Home'
    ))
    fig.add_trace(go.Scatterpolar(
        r=away_values,
        theta=labels,
        fill='toself',
        name='Away'
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