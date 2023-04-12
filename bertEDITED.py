from transformers import InputFeatures, InputExample, BertTokenizer, TFBertForSequenceClassification
import glob
import time
import math
from pathlib import Path
import datasets
import pandas as pd
import tensorflow as tf
from tqdm.notebook import tqdm
import re

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


# def fix_gpu():
#     config = ConfigProto()
#     config.gpu_options.allow_growth = True
#     session = InteractiveSession(config=config)


# fix_gpu()

# def wiki_tsv_to_dataframe():
#     df = pd.read_csv('data/processed_train.tsv_processed/wiki_processed.tsv', sep='\t')
#     df = df[['comment_text', 'binary']]
#     df.columns = ['TEXT', 'TOXIC']
#     df1 = df.iloc[:3200]
#     df2 = df.iloc[3200:6400]

#     return df1, df2

# model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# # TRAINING

# # Creating dataframe for training data
# df, df_test = wiki_tsv_to_dataframe()

# def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN): 
#   train_InputExamples = train.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
#                                                           text_a = x[DATA_COLUMN], 
#                                                           text_b = None,
#                                                           label = x[LABEL_COLUMN]), axis = 1)

#   validation_InputExamples = test.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
#                                                           text_a = x[DATA_COLUMN], 
#                                                           text_b = None,
#                                                           label = x[LABEL_COLUMN]), axis = 1)
  
#   return train_InputExamples, validation_InputExamples

# def convert_examples_to_tf_dataset(examples, tokenizer, max_length=512):
#     features = [] # -> will hold InputFeatures to be converted later

#     for e in examples:
#         # Documentation is really strong for this method, so please take a look at it
#         input_dict = tokenizer.encode_plus(
#             e.text_a,
#             add_special_tokens=True,
#             max_length=max_length, # truncates if len(s) > max_length
#             return_token_type_ids=True,
#             return_attention_mask=True,
#             padding='max_length', # pads to the right by default # CHECK THIS for pad_to_max_length
#             truncation=True
#         )

#         input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
#             input_dict["token_type_ids"], input_dict['attention_mask'])

#         features.append(
#             InputFeatures(
#                 input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
#             )
#         )

#     def gen():
#         for f in features:
#             yield (
#                 {
#                     "input_ids": f.input_ids,
#                     "attention_mask": f.attention_mask,
#                     "token_type_ids": f.token_type_ids,
#                 },
#                 f.label,
#             )

#     return tf.data.Dataset.from_generator(gen, ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
#         (
#             {
#                 "input_ids": tf.TensorShape([None]),
#                 "attention_mask": tf.TensorShape([None]),
#                 "token_type_ids": tf.TensorShape([None]),
#             },
#             tf.TensorShape([]),
#         ),
#     )

# DATA_COLUMN = 'TEXT'
# LABEL_COLUMN = 'TOXIC'

# train_InputExamples, validation_InputExamples = convert_data_to_examples(df, df_test, DATA_COLUMN, LABEL_COLUMN)

# train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
# print(train_data)
# train_data = train_data.shuffle(100).batch(32).repeat(2)

# validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
# validation_data = validation_data.batch(32)

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])
# print(train_data)
# print(validation_data)
# model.fit(train_data, epochs=2, validation_data=validation_data)

# model.save_pretrained("model")


# TESTING + OUTPUT

print("TESTING")

# # Shortens files for testing
# for file in glob.glob('comments_test_processed/*'):
#     with open(file) as f:    
#         lines = f.readlines()

#         filename = file.lstrip('comments_test_processed')
#         filename = filename.rstrip('.txt')
#         print(filename)

#         count = 1
#         while(len(lines) > 250):
#             newLines = lines[:250]
#             lines = lines[250:]

#             output = open('comments_test_processed_split/' + filename + str(count) + ".txt", 'w')
#             for line in newLines:
#                 output.write(line)

#             count += 1
        
#         output = open('comments_test_processed_split/' + filename + str(count) + ".txt", 'w')
#         for line in lines:
#             output.write(line)

model = TFBertForSequenceClassification.from_pretrained("model")

# Iterates through subreddit comment files
for file in glob.glob('extra/*'):
    with open(file) as f:    
        lines = f.readlines()

        filename = file.lstrip('extra/')

        print(filename)

        # BERT tokenize here
        tf_batch = tokenizer(lines, padding=True, truncation=True, return_tensors='tf')

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



