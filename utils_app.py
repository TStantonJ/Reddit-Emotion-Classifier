
import  streamlit as st
import os
import glob
import pandas as pd
from transformers import TFBertForSequenceClassification, BertTokenizer
#from Model_ForGIT6_model import tokenize_data
from suggestions import get_data_from_source, split_data, tokenize_tensorize_data, train_model

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
        # split data
        data_split = split_data(data_raw[1],data_raw[2],0.2)
        st.session_state.trainLabelData = data_split[2]
        st.session_state.testLabelData = data_split[3]
        # Tokenize
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        st.session_state.tokenizedData = tokenize_tensorize_data(tokenizer,data_split[0],data_split[1])
        
    



def model_tab():
    st.session_state.modelHyperParams= {}
    st.session_state.modelHyperParams['staircase'] = True
    # Model tabs to either use preexisiting models or train new models
    modelTab1, modelTab2= st.tabs(["Pre-Trained Models", " Train New Model",])


    # Use exisitng model tab
    with modelTab1:
        
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
                    st.write(f"Currently selected model: {st.session_state.model}")
            except KeyError:
                st.session_state.day = model_list[0]
                
        # Allow for selection of model
        selected_model = st.selectbox(
            ("Select a Model"), model_list, key="model", index=None, on_change=update_model_tab)

    # Train new model tab
    with modelTab2:
        st.write("Plae Holder")
        trainModelOption_selectbox = st.selectbox(
            'Select Ddta from Pre-Loaded sources',
            ('Retrain TFBertForSequenceClassification', 'Model B', 'Model C'),
            index=None,
            placeholder="Select model...",)

        if trainModelOption_selectbox == 'Retrain TFBertForSequenceClassification':
            st.session_state.newModelType = "TFBERT"

        # Hyper param options
        trainModelcol1, trainModelcol2 = st.columns(2)
        with trainModelcol1:
            if st.checkbox("Scheduler Staircase",value=True):
                st.session_state.modelHyperParams['staircase'] = False
            else:
                st.session_state.modelHyperParams['staircase'] = True
            batch_size_input =st.text_input(
                "Select a batchsize",
                "128",
            )
            initial_learning_rate_input =st.text_input(
                "Select an inital learning rate",
                "0.0001",
            )
        with trainModelcol2:
            decay_steps_input =st.text_input(
                "Select decay steps",
                "1400",
            )
            decay_rate_input =st.text_input(
                "Select decay rate",
                "0.5",
            )
            epochs_input =st.text_input(
                "Select epoch",
                "15",
            )
        
        if st.button('train'):
            # lock in hyper params
            st.session_state.modelHyperParams['batch_size'] = batch_size_input
            st.session_state.modelHyperParams['initial_learning_rate'] = initial_learning_rate_input
            st.session_state.modelHyperParams['decay_steps'] = decay_steps_input
            st.session_state.modelHyperParams['decay_rate'] = decay_rate_input
            st.session_state.modelHyperParams['epochs'] = epochs_input

            # Transfer input data
            st.session_state.modelHyperParams['input_ids_train'] =  st.session_state.tokenizedData[0]
            st.session_state.modelHyperParams['attention_masks_train'] =  st.session_state.tokenizedData[2]
            st.session_state.modelHyperParams['labels_train'] =  st.session_state.trainLabelData
            #
            model, history = train_model(st.session_state.newModelType, **st.session_state.modelHyperParams)





        

    

def analysis_tab():
    st.subheader("Analysis")

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)
