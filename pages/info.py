import streamlit as st
import pandas as pd
import pandas as pd
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import streamlit as st

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data

st.title("Info")
st.header("References")

st.header("Input structure")
st.write("The input dataset must be a .csv o .xlsx file with one of the following two structures.")
st.write("**Only clinopyroxene dataset:**")
df = pd.read_excel('pages/Example_input.xlsx')
st.table(df)

df_xlsx = to_excel(df)
st.download_button(label='Download an empty form here!',
                                data=df ,
                                file_name= 'df_test.xlsx')


