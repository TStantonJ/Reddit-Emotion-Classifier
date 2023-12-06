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

def split_data(texts, labels, test_size_in):
    print(test_size_in)
    texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=test_size_in)
    return (texts_train, texts_test, labels_train, labels_test)

# Cmbination tokenizer/tensorizer that can process both training data and prediction data
def tokenize_tensorize_data(tokenizer, texts_train, texts_test=None):
    # If not testing material is given, it is assumed this prediction data
    if texts_test is None:
        encoded_dict = tokenizer.encode_plus(
            texts_train,                      
            add_special_tokens = True,
            max_length = 128,           
            padding = 'max_length',
            return_attention_mask = True,   
            return_tensors = 'tf',
            truncation=True)
        
        input_ids = tf.stack([encoded_dict['input_ids'][0]], axis=0)
        attention_mask = tf.stack([encoded_dict['attention_mask'][0]], axis=0)
        return (input_ids, attention_mask)
    
    # If testing matrial is given, it is assumed this is training data
    else:
        # Tokenize
        input_ids_train = []
        input_ids_test = []
        attention_masks_train = []
        attention_masks_test = []

        for text in texts_train:
            # Tokenize train
            encoded_dict = tokenizer.encode_plus(text, # input texts                     
                                add_special_tokens = True, # add [CLS] at the start, [SEP] at the end
                                max_length = 128, # if input text is longer, then it gets truncated
                                padding = 'max_length', # if input text is shorter, then it gets padded to 128
                                return_attention_mask = True,   
                                return_tensors = 'tf',
                                truncation=True)

            input_ids_train.append(encoded_dict['input_ids'][0]) 
            attention_masks_train.append(encoded_dict['attention_mask'][0])

        for text in texts_test:
            # Tokenize test
            encoded_dict = tokenizer.encode_plus(text,                      
                                add_special_tokens = True,
                                max_length = 128,           
                                padding = 'max_length',
                                return_attention_mask = True,   
                                return_tensors = 'tf',
                                truncation=True)  

            input_ids_test.append(encoded_dict['input_ids'][0])
            attention_masks_test.append(encoded_dict['attention_mask'][0])

        # Convert to tensors
        input_ids_train = tf.stack(input_ids_train, axis=0)
        input_ids_test = tf.stack(input_ids_test, axis=0)

        attention_masks_train = tf.stack(attention_masks_train, axis=0)  
        attention_masks_test = tf.stack(attention_masks_test, axis=0)

        return (input_ids_train, input_ids_test, attention_masks_train, attention_masks_test)

def train_model(model_name, **kwargs):
    if model_name == "TFBERT":
        #Unpack data kwargs
        input_ids_train = kwargs['input_ids_train']
        attention_masks_train = kwargs['attention_masks_train']
        labels_train = kwargs['labels_train']

        #Unpack parameter kwargs
        batch_size = kwargs['batch_size']
        initial_learning_rate = kwargs['initial_learning_rate']
        decay_steps = kwargs['decay_steps']
        decay_rate = kwargs['decay_rate']
        staircase = kwargs['staircase']
        epochs = kwargs['epochs']

        model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

        # Learning rate schedule
        initial_learning_rate = 0.0001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase)
        optimizer = Adam(learning_rate=lr_schedule)

        # loss function
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # model compilation
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        # early stopping, checkpoint to get best parameters
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, min_delta=.001, restore_best_weights=True)
        
        # model training
        history = model.fit([input_ids_train, attention_masks_train], 
                            labels_train, 
                            batch_size=batch_size, 
                            epochs=epochs, 
                            validation_split=0.2,
                            callbacks=[early_stopping])
        
        return(model, history)
        