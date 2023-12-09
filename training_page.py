import  streamlit as st
import re
import glob
import pandas as pd
from transformers import TFBertForSequenceClassification, BertTokenizer
from suggestions import  train_model
from datacode import get_data_from_source, split_data, tokenize_tensorize_data

def train_side_bar():
    with st.sidebar:
        if st.button('Reset'):
            # Clear the chat history
            st.session_state.messages = []

# Function taht controls the data selection tabs
# TODO: add current selected data in tab header
def train_data_tabs():
    # Initalize buttons that need it
    st.session_state.butTokenizeDsabled = True

    # Source select for preloaded data
    dataSource = pd.DataFrame()
    dataOption_selectbox = st.selectbox(
    'Select training data from available data sets',
    ('Merged Reddit Data', 'Hugging Face Twitter Data', 'Reddit Source C'),
    index=None,
    placeholder="Select data source...",)

    if dataOption_selectbox == 'Merged Reddit Data':
        dataSource = pd.read_csv('preloadedData/merged_reddit_data.csv')
        st.session_state.dataSource = dataSource
        st.session_state.labels = [i for i in dataSource.columns]
        st.session_state.text = [i for i in dataSource.columns]
        st.session_state.sampleSizeMax = len(dataSource)
        st.session_state.butTokenizeDsabled = False
    elif dataOption_selectbox == 'Hugging Face Twitter Data':
        dataSource = pd.read_json('hug_data.jsonl', lines=True)
        st.session_state.labels = [i for i in dataSource.columns]
        st.session_state.text = [i for i in dataSource.columns]
        st.session_state.dataSource = dataSource
        st.session_state.labels = [i for i in dataSource.columns]
        st.session_state.text = [i for i in dataSource.columns]
        st.session_state.sampleSizeMax = len(dataSource)
        st.session_state.butTokenizeDsabled = False
    elif dataOption_selectbox == 'Reddit Source C':
        pass
    
    labelColumn, textColumn = st.columns(2)
    with textColumn:
        train_text_selection = st.selectbox(
        ("Select a column as text"), st.session_state.text, index=None)
        st.session_state.choosenText = train_text_selection
    with labelColumn:
        # Allow for selection of label and text
        train_label_selection = st.selectbox(
            ("Select a column as label"), st.session_state.labels, index=None)
        st.session_state.choosenLabel = train_label_selection
    
    train_sampleSize_selection =  st.number_input("Select amount of text you want to train on"
                                                ,max_value=st.session_state.sampleSizeMax)
    st.session_state.sampleSize = train_sampleSize_selection
    
    # Display data
    st.dataframe(dataSource, use_container_width=True)

    
    if st.button('Tokenize Data', key='but_tokenize', disabled=st.session_state.butTokenizeDsabled):
        # Format raw data
        data_raw = get_data_from_source(st.session_state.dataSource, st.session_state.sampleSize, st.session_state.choosenLabel,st.session_state.choosenText)
        # split data
        data_split = split_data(data_raw[1],data_raw[2],0.2)
        st.session_state.trainLabelData = data_split[2]
        st.session_state.testLabelData = data_split[3]
        # Tokenize
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        st.session_state.tokenizedData = tokenize_tensorize_data(tokenizer,data_split[0],data_split[1])

def train_model_tab():
    st.session_state.modelHyperParams= {}
    st.session_state.modelHyperParams['staircase'] = True

    # Model tabs to either use preexisiting models or train new models
    modelTab1, modelTab2= st.tabs(["Test already trained models", " Train New Model",])

    # Use exisitng model tab
    with modelTab1:
        # Find current seletion of models
        train_model_files = sorted([ x for x in glob.glob1("content/models", "*") if re.search('model', x)])
        train_model_list = [f"{x}" for x in train_model_files]
                
        # Allow for selection of model
        train_selected_model = st.selectbox(
            ("Select a Model"), train_model_list, index=None)
        st.session_state.model = train_selected_model
        st.write(st.session_state.model)

        
        
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
        if st.checkbox("Scheduler Staircase",value=True):
            st.session_state.modelHyperParams['staircase'] = False
        else:
            st.session_state.modelHyperParams['staircase'] = True
        trainModelcol1, trainModelcol2 = st.columns(2)
        with trainModelcol1:
            
            new_model_name =st.text_input(
                "Select an name for your model",
                "unamed",
            )
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
            t.session_state.modelHyperParams['new_model_name'] =  new_model_name
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





        

    

def train_analysis_tab():
    st.subheader("Analysis")
