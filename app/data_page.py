import streamlit as st
import pickle
import numpy as np
import pandas as pd
from nba_logos import logos
from predict_page import get_dir_path
import stat_viz as stats

def show_data_viz():
    st.markdown(f"""
    # NBA Game Exploration
    """)

    csv_path = get_dir_path("result.csv")
    data = pd.read_csv(csv_path, parse_dates=['date'])

    # Show additional visualizations
    selected_metrics = ['fgpct', '3ppct', 'ftpct']
    labels = ['Field Goal %', '3-Point %', 'Free Throw %']
    #stats.create_dashboard(home_stats, away_stats, selected_metrics, labels)

    df = data

    # Available columns to select from
    available_cols = df.columns.tolist()

    # Multiselect for columns, with 'team' always selected
    selected_cols = st.multiselect(
        'Select the columns to display',
        options=available_cols,
        default='fgpct'
    )

    # Ensure 'team' is always selected
    selected_cols.insert(0, 'team')

    # Check if user has selected at least one stat column
    if len(selected_cols) == 1:
        st.error("Please select at least one statistic column.")
    else:
        # Filter the dataframe based on selected columns
        filtered_df = df[selected_cols]

        st.write("Filtered Dataframe:")
        st.write(filtered_df)

        # If you want to take further actions, such as displaying a plot based on the filtered data
        if st.button("Show Plot"):
            st.bar_chart(filtered_df.set_index('team'))