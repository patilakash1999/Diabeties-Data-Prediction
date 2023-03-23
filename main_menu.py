import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_option_menu import option_menu
df=pd.read_csv('Diabetes.csv')
st.sidebar.header("Welcome")

with st.sidebar:
    selected=option_menu(
        menu_title="Main Menu", #required
        options=["Home","Dataset","Graph","Accuracy"], #required


    )

    
if selected=="Home":
    st.title(f"WELCOME")