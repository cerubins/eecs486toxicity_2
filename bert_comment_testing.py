#!/usr/bin/python3
'''
bert_comment_testing.py

Written by Carlos Rubins & Ranya Liu for EECS 486

General design followed from the second half of Orhan G. Yalcin's article:
https://towardsdatascience.com/sentiment-analysis-in-10-minutes-with-bert-and-hugging-face-294e8a04b671

Many thanks to Orhan for making a very easy to use guide.

How to run:
% python3 bert_comment_testing.py

You will need to install a lot of different python libraries for this.
Tensorflow itself is about 0.5 GB so be warned.

General Structure:
This script uses the fine-tuned model from bert_wikipedia_training.py and creates files that
have the comment lines accompanied by the model's estimate if the comment is toxic or not.
1) Load model made from bert_wikipedia_training.py
2) Split the comments into bite-sized chunks to avoid a memory overflow
3) Iterate through the split files and create a label for each comment
4) Output to output/ folder. This will then be used by analysis.py
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

# Loads finetuned BERT model and tokenizes with training tokenizer
model = TFBertForSequenceClassification.from_pretrained("model")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Shortens files for testing
for file in glob.glob('comments_test_processed/*'):
    with open(file) as f:    
        lines = f.readlines()

        filename = file.lstrip('comments_test_processed')
        filename = filename.rstrip('.txt')
        print(filename)

        # Splits each subreddit's comments into multiple files of 250 lines
        # eg: soccer_processed1.txt, soccer_processed2.txt, etc.....
        count = 1
        while(len(lines) > 250):
            newLines = lines[:250]
            lines = lines[250:]

            # Outputs file of 250 comments
            output = open('comments_test_processed_split/' + filename + str(count) + ".txt", 'w')
            for line in newLines:
                output.write(line)

            count += 1
        
        # Outputs remaining comments
        output = open('comments_test_processed_split/' + filename + str(count) + ".txt", 'w')
        for line in lines:
            output.write(line)

# Iterates through subreddit comment files
for file in sorted(glob.glob('thing/*')):
    with open(file) as f:
        lines = f.readlines()

        filename = file.removeprefix('thing/')

        print(filename)

        # BERT tokenize here
        tf_batch = tokenizer(lines, max_length=128, padding=True, truncation=True, return_tensors='tf')

        # Run the model on the list
        tf_outputs = model(tf_batch)

        # Runs softmax to get predictions
        tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)

        # Label predictions as toxic and non-toxic
        labels = ['Non-Toxic','Toxic']
        
        # Max of predictions gets labelled
        label = tf.argmax(tf_predictions, axis=1)
        label = label.numpy()

        # Prints predictions into output file
        f = open('output/' + filename, 'w')
        for i in range(len(lines)):
            f.write(str(lines[i]) + str(labels[label[i]]) + "\n")



