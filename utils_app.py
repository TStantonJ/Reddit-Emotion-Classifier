
import  streamlit as st
import os
import glob
import pandas as pd
#from Model_ForGIT6_model import tokenize_data
from suggestions import get_data_from_source

def disable(b):
    st.session_state["disabled"] = b

def update_model_tab():
    st.experimental_set_query_params(
        model=st.session_state.model)



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
        dataTab1, dataTab2= st.tabs(["Pre-Loaded Data", " Live Data",])
        with dataTab1:
            # Source select for preloaded data
            dataSource = pd.DataFrame()
            dataOption_selectbox = st.selectbox(
            'Select Ddta from Pre-Loaded sources',
            ('Merged Reddit Data', 'Hugging Face Twitter Data', 'Reddit Source C'),
            index=None,
            placeholder="Select data source...",)

            if dataOption_selectbox == 'Merged Reddit Data':
                dataSource = pd.read_csv('preloadedData/merged_reddit_data.csv')
                st.session_state.dataSource = dataSource
                st.session_state.sampleSize = 14959
                st.session_state.butTokenizeDsabled = False
            elif dataOption_selectbox == 'Hugging Face Twitter Data':
                dataSource = pd.read_json('hug_data.jsonl', lines=True)
                dataSource.rename(columns={'label':'labels'}, inplace=True) # rename label to label_encoded
                st.session_state.dataSource = dataSource
                st.session_state.sampleSize = 14959
                st.session_state.butTokenizeDsabled = False
            elif dataOption_selectbox == 'Reddit Source C':
                pass
            
            # Display of selected data
            st.dataframe(dataSource, use_container_width=True)
        
    with dataTab2:
        st.subheader("On the fly data")

    
    if st.button('Tokenize Data', key='but_tokenize', disabled=st.session_state.butTokenizeDsabled):
        # Format raw data
        data_raw = get_data_from_source(st.session_state.dataSource, st.session_state.sampleSize)
        # Tokenize raw data
        #data_tokenized = 
        #st.session_state.tokenizedData = (tokenize_data)
    



def model_tab():
    # Model tabs to either use preexisiting models or train new models
    modelTab1, modelTab2= st.tabs(["Pre-Trained Models", " Train New Model",])

    # Train new model tab
    with modelTab1:
        st.write("Plae Holder")

    # Use exisitng model tab
    with modelTab2:
        # Find current seletion of models
        query_params = st.experimental_get_query_params()
        model_files = sorted([ x for x in glob.glob1("content/models", "*")])
        model_list = [f"{x}" for x in model_files]

        # List out model selections
        if query_params:
            try:
                selected_model = query_params["model"][0]
                if selected_model in model_list:
                    st.session_state.model = selected_model
                    st.write(st.session_state.model)
            except KeyError:
                st.session_state.day = model_list[0]
                st.write("fail")

        selected_model = st.selectbox(
            ("Select a Model"), model_list, key="model", index=None, on_change=update_model_tab)#,format_func=format_day)

    

def analysis_tab():
    st.subheader("Analysis")

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)
