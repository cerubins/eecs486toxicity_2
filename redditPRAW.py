#!/usr/bin/python3
'''
redditPRAW.py

Written by Carlos Rubins for EECS 486

How to run:
% python3 redditPRAW.py

General Structure:
This script uses a python library which crawls the reddit API to download the most
recent posts and comments from specified subreddits.
1) Load model made from bert_wikipedia_training.py
2) Split the comments into bite-sized chunks to avoid a memory overflow
3) Iterate through the split files and create a label for each comment
4) Output to output/ folder. This will then be used by analysis.py
'''

from IPython import display
import praw
import glob

def main():
    reddit = praw.Reddit(client_id='tFCJsxRKCFRRvUXspS8_yw',
                         client_secret='iuPB4OxQHXSxpH5U6wRCqi6XeAvJOA',
                         user_agent='TheNameIsAtlas')
    headlines = set()

    subreddits = open("subreddits", "r")

    for file in glob.glob('posts/*'):
        open(file, 'w').close()

    for file in glob.glob('comments/*'):
        open(file, 'w').close()

    for line in subreddits:
        line = line.rstrip('\n')
        for post in reddit.subreddit(line).new(limit=100):
            if post.title not in headlines:
                fposts = open("posts/" + line + ".txt", "a")
                fposts.write(post.title + '\n')
                fposts.close()

                fcomments = open("comments/" + line + ".txt", "a")

                post.comments.replace_more(limit=100, threshold=10)
                for comment in post.comments.list():
                    if comment.body != "" and comment.body != "[deleted]":
                        preprocess = comment.body.replace("\n", " ")
                        fcomments.write(preprocess + "\n")
                
                fcomments.close()

                headlines.add(post.title)
        headlines = set()
    
    subreddits.close()

if __name__ == "__main__":
    main()