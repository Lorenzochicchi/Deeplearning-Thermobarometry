import streamlit as st
import pandas as pd
import pandas as pd
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import streamlit as st
from PIL import Image

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

im = Image.open("logo_noBG.png")
st.set_page_config(
    page_title="Deep Thermobarometer",
    page_icon=im,
    layout="wide"
)


st.title("Info")
st.header("References")
col1, col2,col3, col4 = st.columns(4)

with col1:
    im = Image.open("simone.png")
    st.image(im,use_column_width=True, caption='The mind')
with col2:
    im = Image.open("luca.png")
    st.image(im,use_column_width=True, caption='A Big director')
with col3:
    im = Image.open("duccio.png")
    st.image(im,use_column_width=True, caption='Another big director')
with col4:
    im = Image.open("lorenzo.jpeg")
    st.image(im,use_column_width=True, caption= 'The poor Phd student')

st.header("Input structure")
st.write("The input dataset must be a .xlsx file with the following structure:")
st.write("**Only clinopyroxene dataset:**")
df = pd.read_excel('pages/Example_input.xlsx')
st.table(df)

df_empy = pd.read_excel('pages/Form_input.xlsx') 
df_xlsx = to_excel(df_empy)
st.download_button(label='Download an empty form here!',
                                data=df_xlsx ,
                                file_name= 'Empty_form.xlsx')


