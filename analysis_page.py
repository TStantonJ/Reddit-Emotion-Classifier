import streamlit as st
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

import matplotlib.pyplot as plt

def preprocess_text(self, text):
    # Ensure text is not None
    #if text is None:
    #    return 'none none none none'
    
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9.,;:!?\'\"-]', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    text = re.sub(' +', ' ', text)

    # Lemmatize
    doc = nlp(text)
    text = ' '.join([lemmatizer.lemmatize(token.text) for token in doc])

    return text

def analysis_data_tabs():
    # Initalize buttons that need it
    st.session_state.butTokenizeDisabled = True

    with st.expander("Data Sources",expanded=True):
        #st.subheader("Data Sources")
        dataTab1, dataTab2= st.tabs(["Pre-Loaded Data", " Live Data",])
        with dataTab1:
            # Source select for preloaded data
            dataSource = pd.DataFrame()
            dataOption_selectbox = st.selectbox(
            'Select Data from Pre-Loaded sources',
            (['Reddit post and comments']),
            index=None,
            placeholder="Select data source...",)

            # if dataOption_selectbox == 'Merged Reddit Data':
            #     dataSource = pd.read_csv('preloadedData/merged_reddit_data.csv')
            #     st.session_state.dataSource = dataSource
            # elif dataOption_selectbox == 'Hugging Face Twitter Data':
            #     dataSource = pd.read_json('hug_data.jsonl', lines=True)
            #     dataSource.rename(columns={'label':'labels'}, inplace=True) # rename label to label_encoded
            #     st.session_state.dataSource = dataSource
            if dataOption_selectbox == 'Reddit post and comments':
                dataSource = pd.read_csv('reddit_posts_and_comments.csv', parse_dates = ['Creation Date'])
                st.session_state.dataSource = dataSource
            
            # Display of selected data
            st.dataframe(dataSource, use_container_width=True)
        
    with dataTab2:
        st.subheader("On the fly data")

        commentscolumn, postscolumn = st.columns(2)
        with commentscolumn:
            # button for num of comments
            num_comments = st.number_input('Number of comments', min_value=1, max_value=100, value=3, step=1, format=None, key=None)
        with postscolumn:
            # button for number of posts
            num_posts = st.number_input('Number of posts', min_value=1, max_value=100, value=5, step=1, format=None, key=None)

        # button for subreddit name
        subreddit_name = st.text_input('Subreddit name', value='wallstreetbets', max_chars=None, key=None, type='default')

        timeFiltercolumn, intervalcolumn = st.columns(2)
        with timeFiltercolumn:
            time_filter = st.selectbox('Time filter(Draw from the past ...)', ('day', 'week', 'month', 'year'), index=2, key=None)
        with intervalcolumn:
            # button for interval
            interval = st.selectbox('Interval', ('daily', 'weekly', 'monthly'), index=1, key=None)

        # button for output file
        #output_file = st.text_input('Output file name', value='reddit_posts_and_comments.csv', #max_chars=None, key=None, type='default')

        
        # You can include a button to trigger the scraping process
        if st.button('Fetch Live Data'):
            # inputs: client_id, client_secret, user_agent, num_posts, subreddit_name, interval, top_comments_count, output_file
            df = reddit_scraper('nFKOCvQQEIoW2hFeVG6kfA', 
                                '5BBB4fr-HMPtO8f4jZhle74-fYcDkQ', 
                                'Icy_Process3191', 
                                num_posts=num_posts,
                                subreddit_name=subreddit_name, 
                                time_filter=time_filter, 
                                interval=interval, 
                                top_comments_count=num_comments, 
                                output_file='reddit_posts_and_comments.csv')
            
            #st.write(df)
            
           # Assuming the reddit_scraper function returns a dataframe
            if df is not None and not df.empty:
                st.write(f"First few rows the fetched data (out of {len(df)}):")
                st.dataframe(df.head(), use_container_width=True)
                df.to_csv('output.csv', index=True)

            else:
                st.write("No live data fetched")
                
def analysis_model_tab(): 
  
    model_directory = "content/models"
    all_items = glob.glob(os.path.join(model_directory, "*"))
    analysis_model_files = sorted([x for x in all_items if re.search('model', os.path.basename(x))])


    analysis_model_list = [f"{x}" for x in analysis_model_files]
    analysis_tokenizer_files = sorted([ x for x in glob.glob1("content/models", "*") if re.search('tokenizer', x)])
    analysis_tokenizer_list = [f"{x}" for x in analysis_tokenizer_files]
            
    # Allow for selection of model
    analysis_selected_model = st.selectbox(
        ("Select am Model"), analysis_model_list, index=None)
    st.session_state.model = analysis_selected_model
    if analysis_selected_model:
        if re.search(r'bert', st.session_state.model, re.IGNORECASE) is not None:
            st.session_state.model_name = 'bert'
        elif re.search(r'electra', st.session_state.model, re.IGNORECASE) is not None:
            st.session_state.model_name = 'electra'
        elif re.search(r'roberta', st.session_state.model, re.IGNORECASE) is not None:
            st.session_state.model_name = 'roberta'

    # Allow for selection of model
    analysis_selected_tokenizer = st.selectbox(
        ("Select am tokenizer"), analysis_tokenizer_list, index=None)
    st.session_state.tokenizer = analysis_selected_tokenizer
    st.write([st.session_state.model, st.session_state.tokenizer])

    st.session_state.tokenizer = ''

    if st.button('Apply Model'):
        sentiment = pipeline('sentiment-analysis')
        cwd = os.getcwd()

        #classifier = EmotionClassifier(cwd + '/content/models/' + st.session_state.model, cwd + '/content/models/' +  st.session_state.tokenizer)
        classifier = EmotionClassifier(model_name=st.session_state.model_name, 
                                        model_path=cwd + '/' + st.session_state.model, 
                                        tokenizer_path=''#cwd + '/content/models/' +  st.session_state.tokenizer
                                        )
        print("classifer loaded")
        print(classifier)
        # Arrange Date groups by selected range

        # Predict on each piece of data and store in its date group
        df = pd.read_csv('output.csv')
        sent_scores = []

        # datum_preprocessed = classifier.preprocess_text(text)
        prediction, probs = classifier.predict_emotions(df['Text'].tolist())
        df['Sentiment'] = prediction
        emotion_columns = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
        df[emotion_columns] = probs
        emotion_columns.extend(['Positive','Negative'])
        # Apply sentiment analysis
        df['pos/neg'] = df['Text'].apply(lambda x: sentiment(x, max_length=512)[0]['label'])
        df['pos/neg score'] = df['Text'].apply(lambda x: sentiment(x, max_length=512)[0]['score'])

        df['Positive'] = df.apply(
            lambda x: x['pos/neg score'] if x['pos/neg'] == 'POSITIVE' else 1 - x['pos/neg score'], axis=1)
        df['Negative'] = df.apply(
            lambda x: x['pos/neg score'] if x['pos/neg'] == 'NEGATIVE' else 1 - x['pos/neg score'], axis=1)

        # Compute average for each emotion for each interval
        combined_averages = df.groupby('Interval Number')[emotion_columns].mean()

        # Save the DataFrame
        combined_averages.to_csv('combined_averages.csv')

        st.write(combined_averages)

        fig, axes = plt.subplots(len(combined_averages.columns), 1, figsize=(10, 5 * len(combined_averages.columns)))

        for i, column in enumerate(combined_averages.columns):
            combined_averages[column].plot(ax=axes[i], marker='o', title=column)
            axes[i].set_ylabel('Average Score')
            axes[i].set_xlabel('Interval Number')
            axes[i].invert_xaxis()
            
            # change the x-axis ticks to be the inverse of the interval number
            axes[i].set_xticks(combined_averages.index)
            axes[i].set_xticklabels(combined_averages.index[::-1])


        plt.tight_layout()

        # Use Streamlit's pyplot function to display the figure
        st.pyplot(fig)

        emotion_columns = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))

        # Iterate over each emotion and plot it on the same Axes
        for emotion in emotion_columns:
            combined_averages[emotion].plot(ax=ax, marker='o', label=emotion)

        # Adding title and labels
        ax.set_title('Emotion Scores vs Interval Number')
        ax.set_ylabel('Average Emotion Score')
        ax.set_xlabel('Interval Number')

        # Invert the x-axis and adjust the x-ticks
        ax.invert_xaxis()
        ax.set_xticks(combined_averages.index)
        ax.set_xticklabels(combined_averages.index[::-1])

        # Adding legend to distinguish different emotions
        ax.legend()

        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

        attributes = ['Positive', 'Negative']

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))

        # Iterate over each attribute and plot it on the same Axes
        for attribute in attributes:
            combined_averages[attribute].plot(ax=ax, marker='o', label=attribute)

        # Adding title and labels
        ax.set_title('Positive and Negative Scores vs Interval Number')
        ax.set_ylabel('Average Score (considering proportion)')
        ax.set_xlabel('Interval Number')

        # Invert the x-axis and adjust the x-ticks
        ax.invert_xaxis()
        ax.set_xticks(combined_averages.index)
        ax.set_xticklabels(combined_averages.index[::-1])

        # Adding legend to distinguish between Positive and Negative scores
        ax.legend()

        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

        # average_scores = df.groupby(['Interval Number', 'Sentiment'])['Score'].mean().reset_index()
        # st.write(average_scores)



def reddit_scraper(client_id, client_secret, user_agent, num_posts, subreddit_name, interval, time_filter, top_comments_count, output_file):
    class RedditScraper:
        def __init__(self, client_id, client_secret, user_agent):
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent)

        def fetch_posts(self, num_posts, sub_name, interval):
            subreddit = self.reddit.subreddit(sub_name)
            print(time_filter)
            posts = subreddit.top(time_filter=str(time_filter), limit=num_posts)
            posts_list = list(posts)
            posts_list.sort(key=lambda post: post.created_utc, reverse=True)

            intervals = {
                'daily': timedelta(days=1),
                'weekly': timedelta(weeks=1),
                'monthly': timedelta(weeks=4)}

            end_time = datetime.utcfromtimestamp(posts_list[0].created_utc)
            nested_posts = []
            current_interval_start = end_time
            data = []
            interval_num = 0

            for post in posts_list:
                post_time = datetime.utcfromtimestamp(post.created_utc)

                if post_time < current_interval_start - intervals[interval]:
                    interval_num += 1
                    current_interval_start = post_time

                data.append({
                    'Post/Comment': 'Post',
                    'ID': post.id,
                    'Text': post.title + post.selftext,
                    'Creation Date': datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d'),
                    'Interval Number': interval_num})

            return data, posts_list

        def fetch_comments(self, submission, limit, interval_num):
            submission.comment_sort = 'best'
            submission.comments.replace_more(limit=0)

            return [{'Post/Comment': 'Comment', 'ID': submission.id, 'Text': comment.body,
                     'Creation Date': datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d'),
                     'Interval Number': interval_num} for comment in submission.comments.list()[:limit]]

        def create(self, num_posts, subreddit_name, interval, top_comments_count, output_file):

            data, posts_list = self.fetch_posts(num_posts, subreddit_name, interval)
            interval_nums = [d['Interval Number'] for d in data]

            with ThreadPoolExecutor() as executor:
                comments_list = list(executor.map(lambda p: self.fetch_comments(p[0], top_comments_count, p[1]),
                                                  list(zip(posts_list, interval_nums))))

            data.extend([comment for comment_list in comments_list for comment in comment_list])

            df = pd.DataFrame(data)
            #df.to_csv(output_file, index=True)
            return df

    scraper = RedditScraper(client_id, client_secret, user_agent)
    tmp = scraper.create(num_posts, subreddit_name, interval, top_comments_count, output_file)
    return tmp


def plot_sentiment_scores(average_scores):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(average_scores)), average_scores, color='skyblue')
    plt.xlabel('Interval Number')
    plt.ylabel('Average Sentiment Score')
    plt.title('Average Sentiment Scores per Interval')
    plt.xticks(range(len(average_scores)))
    plt.ylim([0, 1])
    return plt
