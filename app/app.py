import streamlit as st
from predict_page import show_predict_page
from data_page import show_data_viz


page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))

if page == "Predict":
    show_predict_page()
else:
    show_data_viz()