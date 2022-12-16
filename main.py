"""
# My first app

"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import os
#import io
import PIL

def predict(data):
  control = 0 
  if len(data.columns) == 13:
    im_s = 0
  elif len(data.cilumns) == 24:
    im_s = 1
  else:
    print("Number of  dataset columns not allowed")
    control = 1
    return None

  if control ==0:

    for tg in [0,1]:

      if tg == 0:
          directory = 'Pressure_models'  
      else:
          directory = 'Temperature_models'

      targets = ['P (kbar)', 'T (K)']
      target = targets[tg]
      names_targets = ['pressure','temperature']
      names_target = names_targets[tg]

      input_sections = ['only_cpx', 'cpx_and_liq']
      sect = input_sections[im_s]

      with open(directory + '/mod_' + names_target + '_' + sect + '/Global_variable.pickle', 'rb') as handle:
          g = pickle.load(handle)
      N = g['N']
      array_max = g['array_max']

      col = data.columns
      index_col = [col[i] for i in range(0, 2)]
      df1 = data.drop(columns=index_col)

      if tg == 0:
          df_output = pd.DataFrame(
              columns=index_col[:] + ['mean - ' + targets[0], 'std - ' + targets[0], 'mean - ' + targets[1],
                                      'std - ' + targets[1]])

      results = np.zeros((len(df1), N))
      for e in range(N):
          print(e)
          model = tf.keras.models.load_model(
              directory + "/mod_" + names_target + '_' + sect + "/Bootstrap_model_" + str(e) + '.h5')
          results[:, e] = model(df1.values.astype('float32')).numpy().reshape((len(df1),))

      results = results * array_max[0]

      df_output[index_col] = df[index_col]
      df_output['mean - ' + target] = results.mean(axis=1)
      df_output['std - ' + target] = results.std(axis=1)
  return df_output


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def plothist(df):
  targets = ['P (kbar)', 'T (K)']
  for tg in [0,1]:
    x = df['mean - ' + targets[tg]].values.reshape(-1, 1)
    plt.figure()
    plt.hist(df['mean - ' + targets[tg_k]].values,  bins=[i/2 for i in range(-10,21)], density=True, edgecolor='k', color='tab:green',label='hist')
    plt.title('pressure distribution', fontsize=13)
    plt.xlabel('P(kbar)', fontsize=13)
    plt.show()  


    
im = Image.open("D4V.ico")
st.set_page_config(
    page_title="D4V",
    page_icon=im,
    layout="wide",
)

#st.beta_set_page_config(page_title='Deep learning thermobarometer', page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')
st.title("Deeplearning 4 Vulcanoes")
st.header("A deep learning based model to predict temperatures and pressures of vulcanos" )
st.text("The D4V model take as input a dataset of clinopyroxene concentrations..")



uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  filename = uploaded_file.name
  nametuple = os.path.splitext(filename)

  if nametuple[1] == '.csv':
    #read csv
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
  elif nametuple[1] == '.xls' or nametuple[1] == '.xlsx':
    #read xls or xlsx
    df = pd.read_excel(uploaded_file)
    st.dataframe(df)
  else:
    st.warning("File type wrong (you need to upload a csv, xls or xlsx file)") 



if st.button('Starting prediction'):
  df_output = predict(df)
  csv = convert_df(df_output )
  #towrite = io.BytesIO()
  #excel = df.to_excel(towrite, encoding='utf-8', index=False, header=True)

  #st.download_button(
  #    label="Download data as xlsx",
  #    data=excel,
  #    file_name= 'Prediction'+nametuple[0]+'.xlsx',
  #    mime='application/vnd.ms-excel'
  #)
  
  st.download_button(
      label="Download data as csv",
      data=csv,
      file_name= 'Prediction_'+nametuple[0]+'.csv',
      mime='text/csv',
  )
  
  st.write('Predicted values:')
  st.dataframe(df_output)

  if st.button('Plot Histogram'):
    plothist(df_output)
 
  



























