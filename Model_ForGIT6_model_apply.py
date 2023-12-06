import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np
import os
import re
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Load spacy model and stopwords
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

class EmotionClassifier:
    def __init__(self, model_path, tokenizer_path):
        self.model = TFBertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    def preprocess_text(self, text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9.,;:!?\'\"-]', ' ', text)
        text = text.lower()
        text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
        text = re.sub(' +', ' ', text)

        # Lemmatize
        doc = nlp(text)
        text = ' '.join([lemmatizer.lemmatize(token.text) for token in doc])

        return text


    def predict_emotion(self, text):
        # Tokenize the input text
        encoded_dict = self.tokenizer.encode_plus(
            text,                      
            add_special_tokens = True,
            max_length = 128,           
            padding = 'max_length',
            return_attention_mask = True,   
            return_tensors = 'tf',
            truncation=True)

        input_ids = tf.stack([encoded_dict['input_ids'][0]], axis=0)
        attention_mask = tf.stack([encoded_dict['attention_mask'][0]], axis=0)

        # Predict
        logits = self.model.predict([input_ids, attention_mask]).logits
        probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]

        # You can modify this part as per your label names
        emotions = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
        emotion_probs = dict(zip(emotions, probabilities))

        return emotion_probs


# define directory and load previously trained model
cur_dir = '/home/ubuntu/NLP/Project_test/data_model'
os.chdir(cur_dir)



# Example usage
if __name__ == "__main__":
    # Paths to the saved model and tokenizer
    model_path = 'bert_emotion_classifier_reddit'
    tokenizer_path = 'bert_emotion_classifier_tokenizer_reddit'

    classifier = EmotionClassifier(model_path, tokenizer_path)

    # Example text
    text = "Well, his ex wife is like Batman. She’s giving all of her 60bn away"

    # Get prediction
    text = classifier.preprocess_text(text)
    prediction = classifier.predict_emotion(text)
    print("Predicted Emotion Probabilities:", prediction)


cur_dir = '/home/ubuntu/NLP/Project_test'
os.chdir(cur_dir)
