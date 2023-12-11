import json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import TFElectraForSequenceClassification, ElectraTokenizer, ElectraConfig

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import ExponentialDecay


# Function to load and preprocess data from JSON Lines file
def load_data_from_jsonl(filename):
    texts = []
    labels = []

    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, 1):
            try:
                data = json.loads(line)
                texts.append(data['text'])
                labels.append(data['label'])
            except json.JSONDecodeError as e:
                print(f"Error in line {line_number}: {e}")
                break

    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='tf',
            truncation=True
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = tf.concat(input_ids, 0)
    attention_masks = tf.concat(attention_masks, 0)

    labels = np.array(labels)

    labels_series = pd.Series(labels)
    return input_ids, attention_masks, labels, texts, labels_series

# Function to create and compile the ElectraRoberta model
def create_model(num_labels=6):

    config = ElectraConfig.from_pretrained('google/electra-small-discriminator', num_labels=num_labels)
    model = TFElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator', config=config)

    initial_learning_rate = 5e-5
    decay_rate = 0.9
    decay_steps = 1000

    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )


    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


def train_model(model, train_data, validation_data):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3),
        ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, save_format='h5')
    ]

    history = model.fit(
        train_data,
        epochs=10,
        batch_size=128,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1
    )
    return history

# Function for evaluating the model
def evaluate_model(model, validation_data, texts_validation, label_names):

    input_ids_test, attention_masks_test, labels_test = validation_data.values()

    texts_test_series = pd.Series(texts_validation, name='Text')

    y_pred_logits = model.predict({'input_ids': input_ids_test, 'attention_mask': attention_masks_test}).logits
    y_pred_scores = tf.nn.softmax(y_pred_logits, axis=1).numpy()
    y_pred_labels = tf.argmax(y_pred_scores, axis=1).numpy()

    scores_df = pd.DataFrame(y_pred_scores, columns=label_names)
    final_df = pd.concat([texts_test_series, scores_df], axis=1)

    final_df['Overall_Score'] = final_df[label_names].max(axis=1)

    report = classification_report(labels_test, y_pred_labels, target_names=label_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    return final_df, report_df


def load_model():
    return tf.keras.models.load_model('electra_model.h5')


tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')

input_ids, attention_masks, labels, texts, labels_series = load_data_from_jsonl('hug_data.jsonl')

# Check class distribution
class_distribution = labels_series.value_counts(normalize=True) * 100
print("Class Distribution:\n", class_distribution)


if isinstance(input_ids, tf.Tensor):
    input_ids = input_ids.numpy()
if isinstance(attention_masks, tf.Tensor):
    attention_masks = attention_masks.numpy()
if isinstance(labels, tf.Tensor):
    labels = labels.numpy()


train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=2018)
train_masks, validation_masks, texts_train, texts_validation = train_test_split(attention_masks, texts, test_size=0.2, random_state=2018)

print("Dataset Sizes :")
print(len(train_labels),len(validation_labels))


train_inputs = tf.convert_to_tensor(train_inputs)
validation_inputs = tf.convert_to_tensor(validation_inputs)
train_masks = tf.convert_to_tensor(train_masks)
validation_masks = tf.convert_to_tensor(validation_masks)
train_labels = tf.convert_to_tensor(train_labels)
validation_labels = tf.convert_to_tensor(validation_labels)

train_data = {'input_ids': train_inputs, 'attention_mask': train_masks, 'labels': train_labels}
validation_data = {'input_ids': validation_inputs, 'attention_mask': validation_masks, 'labels': validation_labels}

label_names = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']

electra_model = create_model(num_labels=len(label_names))
train_history = train_model(electra_model, train_data, validation_data)
final_df, report_df = evaluate_model(electra_model, validation_data, texts_validation, label_names)


print("Evaluation Scores:")
print(final_df.head())

print("\nClassification Report:")
print(report_df)