import streamlit as st
import pandas as pd

st.title("Info")
st.header("References")

st.header("Input structure")
st.write("The input dataset must be a .csv o .xlsx file with the following two structure:")
st.write("**Only clinopyroxene datase")
df = pd.read_excel('Example_input.xlsx')
st.table(df)
