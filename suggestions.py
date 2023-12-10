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
        

def evaluate_model(model, **kwargs):

    input_ids_test = kwargs['input_ids_test']
    attention_masks_test = kwargs['attention_masks_test']
    labels_test = kwargs['labels_test']
    texts_test= kwargs['texts_test']

    # Model evaluation for scores
    y_pred_logits = model.predict_emotions([input_ids_test, attention_masks_test]).logits
    y_pred_scores = tf.nn.softmax(y_pred_logits, axis=1).numpy()
    y_pred_labels = tf.argmax(y_pred_logits, axis=1).numpy()

    # Creating DataFrame
    texts_test_series = pd.Series(texts_test, name='text')
    scores_df = pd.DataFrame(y_pred_scores, columns=['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise'])
    final_df = pd.concat([texts_test_series, scores_df], axis=1)

    # Adding overall score
    final_df['Overall_Score'] = final_df[['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']].max(axis=1)

    # get classification report
    report = classification_report(labels_test, y_pred_labels, target_names=['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise'], output_dict=True)
    report_df = pd.DataFrame(report).transpose() # reset index needs to  be tested; intended to show labels in the final report_df cvs

    # report to csv
    #report_df.to_csv('hug_data_sample_report.csv', index=False)

    # preview report
    return final_df, report_df