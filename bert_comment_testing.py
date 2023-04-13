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

print("TESTING")

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

# Loads trained BERT model from saved file
model = TFBertForSequenceClassification.from_pretrained("model")

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



