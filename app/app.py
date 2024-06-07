import streamlit as st
from predict_page import show_predict_page
from sports_book_page import show_sports_book_page


page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Sports Book"))

if page == "Predict":
    show_predict_page()
else:
    show_sports_book_page()