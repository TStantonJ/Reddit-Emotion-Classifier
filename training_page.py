import  streamlit as st
import re
import os
import glob
import pandas as pd
from transformers import TFBertForSequenceClassification, BertTokenizer
from suggestions import  train_model, evaluate_model
from datacode import get_data_from_source, split_data, tokenize_tensorize_data
from Model_ForGIT6_model_apply import EmotionClassifier
import streamlit as st
from sklearn.metrics import classification_report
import re
import glob
import pandas as pd
from transformers import TFBertForSequenceClassification, BertTokenizer
from datacode import get_data_from_source, split_data, tokenize_tensorize_data
from Model_ForGIT6_model_apply import*
import statistics
import praw
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import prawcore
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
import pandas as pd
import numpy as np
import os
from collections import Counter
from Model_ForGIT6_model_retrain import preprocess_text, tokenize_data, conv_to_tensor


def train_data_tabs():
    """
    Control process for selecting data when testing or training models.

    Args:
        None

    Returns:
        None
    """

    # Initalization of session_state variables
    st.session_state.butTokenizeDsabled = True
    st.session_state.labels = []
    st.session_state.text = []
    st.session_state.dataSource = pd.DataFrame()
    st.session_state.labels = []
    st.session_state.text = []
    st.session_state.sampleSize = 0
    st.session_state.butTokenizeDsabled = False
    
    
    # Select data source(Modular for optinal future addition of datasets)
    dataSource = pd.DataFrame()
    dataOption_selectbox = st.selectbox(
    'Select training data from available data sets',
    (['Hugging Face Twitter Data']),
    index=None,
    placeholder="Select data source...",)

    if dataOption_selectbox == 'Hugging Face Twitter Data':
        # Assign data to session_state variables
        dataSource = pd.read_json('hug_data.jsonl', lines=True)
        st.session_state.labels = [i for i in dataSource.columns]
        st.session_state.text = [i for i in dataSource.columns]
        st.session_state.dataSource = dataSource
        st.session_state.labels = [i for i in dataSource.columns]
        st.session_state.text = [i for i in dataSource.columns]
        st.session_state.sampleSize = 14959
        st.session_state.butTokenizeDsabled = False
    
        # Select box's for user selection of label and text from dataset
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
    
    # Display choosen data source
    st.dataframe(dataSource, use_container_width=True)

    # Button to tokenize data as well as breakout into testing and training sets in session_state variables
    if st.button('Tokenize Data', key='but_tokenize', disabled=st.session_state.butTokenizeDsabled):
        # Format raw data into label and text sets
        data_raw = get_data_from_source(st.session_state.dataSource, st.session_state.sampleSize, st.session_state.choosenLabel,st.session_state.choosenText)
        
        # Split data into testing and training sets, storing the data in 
        data_split = split_data(data_raw[1],data_raw[2],0.2)
        st.session_state.data_split = data_split
        st.session_state.trainLabelData = data_split[2]
        st.session_state.testLabelData = data_split[3]
        
        # Tokenize the sets and store the sets in the session_state variables
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        st.session_state.tokenizedData = tokenize_tensorize_data(tokenizer,data_split[0],data_split[1])

def train_model_tab():
    """
    Control process for selecting model type when testing or training models.
    As well as testing/traing that model.

    Args:
        None

    Returns:
        None
    """
    st.session_state.modelHyperParams= {}
    st.session_state.testModelHyperParams ={}
    st.session_state.modelHyperParams['staircase'] = True

    # Model tabs to either use preexisiting models or train new models
    modelTab1, modelTab2= st.tabs(["Test already trained models", " Train New Model",])

    # Test exisitng model tab
    with modelTab1:
        # Find current seletion of models
        train_model_files = sorted([ x for x in glob.glob1("content/models", "*") if re.search('model', x)])
        train_model_list = [f"{x}" for x in train_model_files]
                
        # Allow for selection of model
        train_selected_model = st.selectbox(
            ("Select a Model"), train_model_list, index=None)
        st.session_state.model = train_selected_model
        st.write(st.session_state.model)

        if st.button('Test'):
            cwd = os.getcwd()
            classifier = EmotionClassifier(model_name=st.session_state.model_name, 
                                            model_path=cwd + '/content/models/' + st.session_state.model, 
                                            tokenizer_path=''#cwd + '/content/models/' +  st.session_state.tokenizer
                                            )
            print("classifer loaded")
            print(classifier)

            print('preprocessing reddit data')
            df_reddit = pd.read_json('hug_data.jsonl', lines=True)
            df_reddit['text'] = df_reddit['text'].apply(preprocess_text)
            # Arrange Date groups by selected range

            texts_reddit = df_reddit['text'].values
            labels_reddit = df_reddit['label'].values
            texts_test=[]
            input_ids_train, input_ids_test, attention_masks_train, attention_masks_test, classifier.tokenizer = tokenize_data(texts_reddit, texts_test)
            input_ids_train, input_ids_test, attention_masks_train, attention_masks_test = conv_to_tensor(input_ids_train, input_ids_test, attention_masks_train, attention_masks_test)

            final_df, report_df = evaluate_model(classifier, input_ids_train, attention_masks_train, labels_reddit)
            st.write(final_df, report_df)
            #st.write(evaluate_model(classifier,df, **st.session_state.testModelHyperParams))


        
    # Train new model tab
    with modelTab2:
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
                "unnamed",
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
            st.session_state.modelHyperParams['new_model_name'] =  new_model_name
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
