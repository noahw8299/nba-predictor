import streamlit as st
import pickle
import numpy as np
import pandas as pd
from nba_logos import logos

def show_sports_book_page():

    bankroll = 10000

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"Bankroll: {bankroll}")

    with col2:
        st.write("asoeif")



    ok = st.button("BET")

    if ok:
        return