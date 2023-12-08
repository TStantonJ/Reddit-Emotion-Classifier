import  streamlit as st
import re
import glob
import pandas as pd
from transformers import TFBertForSequenceClassification, BertTokenizer
from suggestions import  train_model
from datacode import get_data_from_source, split_data, tokenize_tensorize_data
from Model_ForGIT6_model_apply import*
import statistics

def analysis_data_tabs():
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
            ('Merged Reddit Data', 'Hugging Face Twitter Data', 'Reddit post and comments'),
            index=None,
            placeholder="Select data source...",)

            if dataOption_selectbox == 'Merged Reddit Data':
                dataSource = pd.read_csv('preloadedData/merged_reddit_data.csv')
                st.session_state.dataSource = dataSource
            elif dataOption_selectbox == 'Hugging Face Twitter Data':
                dataSource = pd.read_json('hug_data.jsonl', lines=True)
                dataSource.rename(columns={'label':'labels'}, inplace=True) # rename label to label_encoded
                st.session_state.dataSource = dataSource
            elif dataOption_selectbox == 'Reddit post and comments':
                dataSource = pd.read_csv('reddit_posts_and_comments.csv', parse_dates = ['Creation Date'])
                st.session_state.dataSource = dataSource
            
            # Display of selected data
            st.dataframe(dataSource, use_container_width=True)
        
    with dataTab2:
        st.subheader("On the fly data")

def analysis_model_tab(): 
    # Find current seletion of models
    analysis_model_files = sorted([ x for x in glob.glob1("content/models", "*") if re.search('model', x)])
    analysis_model_list = [f"{x}" for x in analysis_model_files]
    analysis_tokenizer_files = sorted([ x for x in glob.glob1("content/models", "*") if re.search('tokenizer', x)])
    analysis_tokenizer_list = [f"{x}" for x in analysis_tokenizer_files]
            
    # Allow for selection of model
    analysis_selected_model = st.selectbox(
        ("Select am Model"), analysis_model_list, index=None)
    st.session_state.model = analysis_selected_model

    # Allow for selection of model
    analysis_selected_tokenizer = st.selectbox(
        ("Select am tokenizer"), analysis_tokenizer_list, index=None)
    st.session_state.tokenizer = analysis_selected_tokenizer
    st.write([st.session_state.model, st.session_state.tokenizer])

    if st.button('Apply Model'):
        cwd = os.getcwd()
        classifier = EmotionClassifier(cwd + '/content/models/' + st.session_state.model, cwd + '/content/models/' +  st.session_state.tokenizer)

        # Arrange Date groups by selected range
        grouped_data = arrange_data(st.session_state.dataSource, 'm')

        # Predict on each piece of data and store in its date group
        st.write(grouped_data[0])
        sent_scores = []
        for time_period in range(len(grouped_data)):
            sent_scores.append([])
            for _,datum in grouped_data[time_period].iterrows():
                #print(datum['Text'])
                datum_text = datum['Text']
                datum_preprocessed =classifier.preprocess_text(datum_text)
                prediction = classifier.predict_emotion(datum_preprocessed)
                sent_scores[time_period].append(prediction)

        # Average date groups
        for i in range(len(sent_scores)):
            average_holder = []
            for j in range(len(sent_scores[i])):
                for k in range(len(sent_scores[i][j])):
                    average_holder[k].append(sent_scores[i][j][k])
            
            for emotion in range(len(average_holder)):
                average_holder[emotion] = statistics.mean(average_holder[emotion])

        # Write averages for now
        st.write(average_holder)

def arrange_data(df, splitBy):
    if splitBy == 'd':
        pass
    elif splitBy == 'm':
        g = df.groupby(pd.Grouper(key='Creation Date', freq='M'))
        groups = [group for _,group in g]
        return groups
    elif splitBy == 'y':
        pass