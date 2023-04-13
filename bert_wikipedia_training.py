#!/usr/bin/python3
'''
bert_wikipedia_training.py

Written by Carlos Rubins for EECS 486
Help from Andrew Mastruserio in using pandas dataframes

General design followed from Orhan G. Yalcin's article:
https://towardsdatascience.com/sentiment-analysis-in-10-minutes-with-bert-and-hugging-face-294e8a04b671

Many thanks to Orhan for making a very easy to use guide.

How to run:
% python3 bert_wikipedia_training.py

You will need to install a lot of different python libraries for this.
Tensorflow itself is about 0.5 GB so be warned.

General Structure:
This script downloads a pre-trained model from Bert and fine-tunes it using our wikipedia dataset.
1) Convert data found in data/wiki_pre_processed.tsv into pandas dataframes
2) Convert data frames into InputExamples
3) Convert InputExamples into tf Dataset that Bert can use
4) Configure pretrained text classification model
5) Fine-tune model using wikipedia dataset (this step will take about 8 hours at the moment, may differ if amount of data is increased)
6) Save model to model folder.
'''
from transformers import InputFeatures, InputExample, BertTokenizer, TFBertForSequenceClassification
import glob
import time
import math
from pathlib import Path
import datasets
import pandas as pd
import tensorflow as tf
from tqdm.notebook import tqdm

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# Fixes an error with tensorflow on some computers
# Specifically if you are using cuda/GPU processing
def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()

# Take information from the wiki dataset and convert them into
# Pandas Dataframes for usage.
def wiki_tsv_to_dataframe():
    df = pd.read_csv('data/wiki_pre_processed.tsv', sep='\t')
    df = df[['comment_text', 'binary']]
    df.columns = ['TEXT', 'TOXIC']
    df1 = df.iloc[:3200]
    df2 = df.iloc[3200:6400]

    return df1, df2

# Declare our model and our tokenizer.
# We will use pretrained models and fine-tune them for our usage
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Creating dataframe for training data
df, df_test = wiki_tsv_to_dataframe()

# Take our Dataframes and convert rows into Input Examples
def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN): 
  train_InputExamples = train.apply(lambda x: InputExample(guid=None,
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)

  validation_InputExamples = test.apply(lambda x: InputExample(guid=None,
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)
  
  return train_InputExamples, validation_InputExamples

# Take Input Example objects and create Bert-readable input
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=512):
    # Holds Input features
    features = []

    for e in examples:

        # Encode tokenization for BERT usage
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            padding='max_length', # pads them to all be the same size
            truncation=True # Works with max_length to cut down to size
        )

        # split features that we want into different dictionaries for usage in generator function
        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    # Generate the bert-readable inputs
    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    # Return dataset from method using our "gen()" function
    return tf.data.Dataset.from_generator(gen, ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )

# Declare our column names
DATA_COLUMN = 'TEXT'
LABEL_COLUMN = 'TOXIC'

# Get InputExamples to convert into datasets
train_InputExamples, validation_InputExamples = convert_data_to_examples(df, df_test, DATA_COLUMN, LABEL_COLUMN)

# Input datasets and convert them into tensorflow datasets
train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)

# Create batches of 32 for testing
train_data = train_data.shuffle(100).batch(32).repeat(2)

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)

# Configure the model to have a small learning rate, due to our small batch size
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

# Fine-tune the model from before
model.fit(train_data, epochs=2, validation_data=validation_data)

# Save the model so we don't have the computer crash :)
model.save_pretrained("model")