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

def train_model(model_name, **kwargs):
    if model_name == "TFBERT":
        #Unpack data kwargs
        input_ids_train = kwargs['input_ids_train']
        attention_masks_train = kwargs['attention_masks_train']
        labels_train = kwargs['labels_train']

        #Unpack parameter kwargs
        new_model_name = kwargs['new_model_name']
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
        
        model.save_pretrained(os.path.join(new_model_name+'_model'))
        #tokenizer.save_pretrained(os.path.join('bert_emotion_classifier_tokenizer_reddit'))
        
        return(model, history)
        