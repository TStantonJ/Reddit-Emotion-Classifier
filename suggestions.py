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
import nltk
import spacy
from nltk.stem import WordNetLemmatizer

# Load spacy model and stopwords
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


# Rename this to format data!
# Plus, import or define tokenize_data and conv_to_tensor
def get_data(sample_size):
    # load data
    df = pd.read_json('hug_data.jsonl', lines=True)
    df.rename(columns={'label':'labels'}, inplace=True) # rename label to label_encoded

    # get subset of df for testing/debugging/development (CHANGE THIS IN THE FUTURE)
    df = df.groupby('labels').apply(lambda x: x.sample(n=sample_size)).reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True) # shuffle df

    texts = df['text'].values
    labels = df['labels'].values

    return df, texts, labels


# Preprocess text function
def preprocess_text(text):
    
    # Remove links
    text = re.sub(r'http\S+', '', text)

    # Remove non-alphanumeric and non puncuation characters
    text = re.sub(r'[^a-zA-Z0-9.,;:!?\'\"-]', ' ', text)

    # Lowercase
    text = text.lower()

    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])

    # Remove extra spaces
    #text = re.sub(' +', ' ', text)

    # Lemmatize
    doc = nlp(text)
    text = ' '.join([lemmatizer.lemmatize(token.text) for token in doc])

    return text

def tokenize_data(texts_train, texts_test):
    # load pretrained tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

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

    return input_ids_train, input_ids_test, attention_masks_train, attention_masks_test, tokenizer

def conv_to_tensor(input_ids_train, input_ids_test, attention_masks_train, attention_masks_test,):
    # Convert to tensors
    input_ids_train = tf.stack(input_ids_train, axis=0)
    input_ids_test = tf.stack(input_ids_test, axis=0)

    attention_masks_train = tf.stack(attention_masks_train, axis=0)  
    attention_masks_test = tf.stack(attention_masks_test, axis=0)
    
    return input_ids_train, input_ids_test, attention_masks_train, attention_masks_test

def train_model(model_name, **kwargs):
    if model_name == "TFBERT":

        #Unpack parameter kwargs
        new_model_name = kwargs['new_model_name']
        batch_size = kwargs['batch_size']
        initial_learning_rate = kwargs['initial_learning_rate']
        decay_steps = kwargs['decay_steps']
        decay_rate = kwargs['decay_rate']
        staircase = kwargs['staircase']
        epochs = kwargs['epochs']

        # define directory and load previously trained model
        cur_dir = os.getcwd()
        os.chdir(cur_dir)

        model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
        tokenizer = BertTokenizer.from_pretrained(cur_dir + "/content/models/bert_emotion_classifier_model_reddit")


        cur_dir = cur_dir + "/content/models"
        os.chdir(cur_dir)

        # Load the saved model
        model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
        
        df = pd.read_json('hug_data.jsonl', lines=True)
        df.rename(columns={'label':'labels'}, inplace=True) # rename label to label_encoded

        # get subset of df for testing/debugging/development (CHANGE THIS IN THE FUTURE)
        df = df.groupby('labels').apply(lambda x: x.sample(n=14959)).reset_index(drop=True)
        df = df.sample(frac=1).reset_index(drop=True) # shuffle df

        texts = df['text'].values
        labels = df['labels'].values

        # Split, tokenize, and convert to tensors
        print('splitting')
        texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2)
        input_ids_train, input_ids_test, attention_masks_train, attention_masks_test, tokenizer = tokenize_data(texts_train, texts_test)
        input_ids_train, input_ids_test, attention_masks_train, attention_masks_test = conv_to_tensor(input_ids_train, input_ids_test, attention_masks_train, attention_masks_test)

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
        

def evaluate_model(model,df, **kwargs):

    input_ids_test = kwargs['input_ids_test']
    attention_masks_test = kwargs['attention_masks_test']
    labels_test = kwargs['labels_test']
    texts_test= kwargs['texts_test']
    print(len(texts_test))

    print(attention_masks_test)
    y_pred_logits = model.predict_emotions([input_ids_test, attention_masks_test]).logits
    y_pred_scores = tf.nn.softmax(y_pred_logits, axis=1).numpy()
    y_pred_labels = tf.argmax(y_pred_logits, axis=1).numpy()

    # Creating DataFrame
    texts_test_series = pd.Series(texts_test, name='Text')
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

