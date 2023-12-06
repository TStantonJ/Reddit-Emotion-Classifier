
import  streamlit as st
import os
import pandas as pd
from Model_ForGIT import *

def disable(b):
    st.session_state["disabled"] = b

def side_bar():
    with st.sidebar:
        if st.button('Reset'):
            # Clear the chat history
            st.session_state.messages = []

# Function taht controls the data selection tabs
# TODO: add current selected data in tab header
def data_tabs():
    # Initalize buttons that need it
    st.session_state.butTokenizeDsabled = True

    with st.expander("Data Sources",expanded=True):
        #st.subheader("Data Sources")
        tab1, tab2= st.tabs(["Pre-Loaded Data", " Live Data",])
        with tab1:
            # Source select for preloaded data
            dataSource = pd.DataFrame()
            dataOption_selectbox = st.selectbox(
            'Select Ddta from Pre-Loaded sources',
            ('Reddit Source A', 'Reddit Source B', 'Reddit Source C'),
            index=None,
            placeholder="Select data source...",)

            if dataOption_selectbox == 'Reddit Source A':
                dataSource = pd.read_csv('preloadedData/merged_reddit_data.csv')
                st.session_state.dataSource = dataSource
                st.session_state.butTokenizeDsabled = False
            elif dataOption_selectbox == 'Reddit Source B':
                pass
            elif dataOption_selectbox == 'Reddit Source C':
                pass
            
            # Display of selected data
            st.dataframe(dataSource, use_container_width=True)
        
    with tab2:
        st.subheader("A tab with the data")

    
    #if st.button('Tokenize Data', key='but_tokenize', disabled=st.session_state.butTokenizeDsabled, style="background-color: #DD3300;"):
    #    st.session_state.tokenizedData = Mo
    



def model_tab():
    st.subheader("Model Sources")

def analysis_tab():
    st.subheader("Analysis")

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)
