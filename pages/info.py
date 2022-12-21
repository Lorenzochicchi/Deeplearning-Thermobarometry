import streamlit as st
import pandas as pd

st.title("Info")
st.header("References")

st.header("Input structure")
st.write("The input dataset must be a .csv o .xlsx file with one of the following two structures.")
st.write("**Only clinopyroxene dataset:**")
df = pd.read_excel('pages/Example_input.xlsx')
st.table(df)
