import  streamlit as st
import re
import glob
import pandas as pd
from transformers import TFBertForSequenceClassification, BertTokenizer
from suggestions import  train_model
from datacode import get_data_from_source, split_data, tokenize_tensorize_data
from Model_ForGIT6_model_apply import*

def analysis_model_tab(): 
    # Find current seletion of models
    analysis_model_files = sorted([ x for x in glob.glob1("content/models", "*") if re.search('model', x)])
    analysis_model_list = [f"{x}" for x in analysis_model_files]
    analysis_tokenizer_files = sorted([ x for x in glob.glob1("content/models", "*") if re.search('tokenizer', x)])
    analysis_tokenizer_list = [f"{x}" for x in analysis_tokenizer_files]
            
    # Allow for selection of model
    analysis_selected_model = st.selectbox(
        ("Select am Model"), analysis_model_list, index=None)#, on_change=update_model_tab)
    st.session_state.model = analysis_selected_model

    # Allow for selection of model
    analysis_selected_tokenizer = st.selectbox(
        ("Select am tokenizer"), analysis_tokenizer_list, index=None)
    st.session_state.tokenizer = analysis_selected_tokenizer
    st.write([st.session_state.model, st.session_state.tokenizer])

    if st.button('Apply Model'):
        cwd = os.getcwd()
        classifier = EmotionClassifier(cwd + '/content/models/' + st.session_state.model, cwd + '/content/models/' +  st.session_state.tokenizer)
        text = "Well, his ex wife is like Batman. She’s giving all of her 60bn away"

        # Get prediction
        text = classifier.preprocess_text(text)
        prediction = classifier.predict_emotion(text)
        st.write("Predicted Emotion Probabilities:", prediction)