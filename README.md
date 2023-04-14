# eecs486toxicity_2

#########################################
List of folders/files and their purposes:
#########################################
comments (folder): contains raw, scrapped comments from all 40 subreddits.
comments_test_processed (folder): contains comments from all 40 subreddits that have been run through 
  reddit_process_simple.py
comments_test_processed_split (folder): contains comments from all 40 subreddits that have been split into
  files of 250 lines. This split was necessary for running BERT smoothly
data (folder): contains raw wikipedia training dataset, as well as the preprocessed wikipedia training dataset (from wikipedia_process_simple.py)
  we used to train our BERT model in bert_wikipedia_training.py
model (folder): contains saved trained BERT model
output (folder): contains BERT predictions for all the preprocessed subreddit data from comments_test_processed_split
output_stitched (folder): contains BERT predictions from output folder, with all split subreddit file results
  combined into one .txt file per subreddit
posts (folder): contains the raw, scraped subreddit posts where we sourced our comments from
process_files (folder): contains dictionaries that assign sentiments to words, used in reddit_process_simply.py
  and wikipedia_process_simple.py
random_samples (folder): contains 100 random samples from output_stiched for each subreddit for human grading
analysis.txt: contains ranked list of subreddits by toxicity rate, as well as their identifier ratios
subreddits: contains list of 40 subreddits we analyzed

analysis.py: 
  input: output folder
  output: output_stitched, analysis.txt, random_samples folder
  description: stitches together files in output, creates random samples for human grading, and finds toxicity data on subreddits

bert_comment_testing.py:
  input: comments_test_processed, model folder
  output: comments_test_processed_split, output folder
  description: splits comments_test_processed into files of 250 lines (put into comments_test_processed_split),
  then runs the saved BERT model to predict toxicity of the preprocessed comments. Predictions are outputed to output folder

bert_wikipedia_testing.py:
  input: data folder
  output: model folder
  description: trains BERT model on preprocessed wikipedia training dataset from the data folder

redditPRAW.py:
  input: subreddits file
  output: posts, comments folders
  description: Scraps given subreddits from the subreddits file and outputs the posts/comments from the posts

reddit_process_simple.py:
  input: comments, process_files
  output: comments_test_processed
  description: Preprocessed raw reddit data, replaces text with their sentiment

wikipedia_process_simple.py:
  input: data, process_files
  output: data 
  description: Preprocessed raw training data, replaces text with their sentiment

###########
How to run:
###########

# Run PRAW, generating posts/ and comments/
% python3 redditPRAW.py

# Place wiki_raw.tsv into folder wikipedia/
# Run the processor on the Wikipedia training set and the Reddit testing set
% python3 reddit_process_simple.py comments/
% python3 wikipedia_process_simple.py wikipedia/

# Place wiki_raw.tsv into data/
# Rename wiki_raw_processed.tsv to wiki_pre_processed.tsv
# Place wiki_pre_processed.tsv into data/
# Train BERT, then test BERT
% python3 bert_wikipedia_training.py
% python3 bert_comment_testing.py

# Run analysis on output
% python3 analysis.py

####################################################################
More details on what each program does is documented within the code
####################################################################
