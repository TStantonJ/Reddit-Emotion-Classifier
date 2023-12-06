import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import TFBertForSequenceClassification, BertTokenizer
from keras.optimizers import Adam
from sklearn.metrics import classification_report
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import re
from nltk.corpus import stopwords
import numpy as np


# Rename this to format data!
def get_data_from_source(data_source, sample_size):
    # load data
    df = data_source
    df.rename(columns={'label':'labels'}, inplace=True) # rename label to label_encoded

    # get subset of df for testing/debugging/development (CHANGE THIS IN THE FUTURE)
    df = df.groupby('labels').apply(lambda x: x.sample(n=sample_size)).reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True) # shuffle df

    texts = df['text'].values
    labels = df['labels'].values

    return (df, texts, labels)
