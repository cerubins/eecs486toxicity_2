'''
analysis.py
Written by Ranya Liu for EECS 486

How to run:
% python3 analysis.py

General Structure:
1) After running BERT on all the preprocessed files and organizing all the outputed
prediction files into subreddit folders (eg: soccer_processed1.txt, soccer_processed2.txt -> soccer folder), 
analysis.py stiches all the files together into a single file. This file should contain all the subreddit's 
comments (after preprocessing) and their predictions from BERT
2) analysis.py then conducts three main functions:
    (a) finding the percentage of toxic comments per subreddit
    (b) finding the percentage of hpositive, positive, neutral, negative, hnegative identifiers per subreddit
    (c) finding average number of words per comment
    (d) generating 100 random samples per subreddit for human accuracy grading
3) analysis.py outputs a file for each subreddit contianing 100 random samples for future grading, as well as an 
analysis.txt file that lists the subreddits from highest toxic comment percentage to lowest, as well as their 
identifier percentages. The last listed number in each subreddit line represents the number of words per comment
'''

import glob
from pathlib import Path
import os
import emoji
import re
import random

# Helper functions for 
def tryint(s):
    try:
        return int(s)
    except:
        return s
# Sorts list alphabetically 
def alphanum_keys(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

# Stitches together separated output files (which were separated for
# smoother BERT analysis), and combines the 250 line
# files into a large file by subreddit
data = Path('output/')
oldpwd = os.getcwd()
for folder in data.iterdir():
    subreddit = folder.name
    comments = []
    # newData = Path(folder)
    # list(newData.iterdir()).sort(key=alphanum_keys)
    # print('output/' + str(folder.name) + '/*')
    
    # Ensures files are read alphabetically (from 1.txt, 2.txt, 3.txt......)
    # instead of (from 1.txt, 10.txt, 11.txt.....)
    li = glob.glob('output/' + str(folder.name) + '/*')
    li.sort(key=alphanum_keys)

    # Totals all comments from files
    for file in li:
        print(file)
        with open(file, 'r', encoding = 'utf-8', errors='ignore') as f:
            comment = f.readlines()
            comments.append(comment)

    # Writes all subreddit comments and their predictions to a single output file
    output = open('output_stitched/' + subreddit, 'w', encoding = 'utf-8', errors='ignore')
    for file in comments:
        output.writelines(file)



# Iterates through all subreddit BERT predictions, and collects data on each
toxicScores = dict()
stats = dict()
for file in glob.glob('output_stitched/*'):
    with open(file, encoding = 'utf-8') as f:
        subreddit = file.lstrip('output_stitched\\')
        lines = f.readlines()

        # finds original comments, before preprocessing to pair
        unprocessed = open('comments_test/' + subreddit + '.txt', encoding = 'utf-8')
        unprocessLines = unprocessed.readlines()

        # initializes data counts
        lineCount = 0
        toxicCount = 0
        wordCount = 0
        index = 0
        pair = ("", "", "")
        comments = []

        # initializes identifier counts
        hpos = 0
        pos = 0
        neu= 0
        neg = 0
        hneg = 0

        for line in lines: 
            # increments number of labeled toxic comments
            if lineCount % 2 == 1 and 'Non' not in line:
                toxicCount += 1
            
            # stores comments 
            if lineCount % 2 == 0:
                # stores preprocessed comment
                update = list(pair)
                update[0] = line

                # finds/stores associated original comment
                update[1] = unprocessLines[index]
                index += 1
                pair = tuple(update)

                # counts number of hpositive/positive/neutral/negative/hnegative
                # word identifiers
                hpos += line.count('hpositive')
                pos += line.count('positive')
                neu += line.count('neutral')
                neg += line.count('negative')
                hneg += line.count('hnegative')

            else:
                # stores tuple of (preprocessed comment, original comment, prediction)
                update = list(pair)
                update[2] = line
                pair = tuple(update)
                comments.append(pair)

            lineCount += 1
            wordCount += len(line.split())

        # Outputs 100 random samples for "hand-grading"
        output = open('random_samples/' + subreddit + ".txt", 'w', encoding = 'utf-8', errors='ignore')
        samples = random.sample(comments, 100)
        for sample in samples:
            hold = list(sample)
            output.write(hold[0])
            output.write("\n")
            output.write(hold[1])
            output.write("\n")
            output.write(hold[2])
            output.write("\n\n\n\n\n")
        
        toxicScores[subreddit] = toxicCount/len(lines)
        
        # stores word identifier ratios for each subreddit
        identifiers = hpos + pos + neg + hneg
        hpos /= identifiers
        pos /= identifiers
        neg /= identifiers
        hneg /= identifiers
        stats[subreddit] = (hpos, pos, neg, hneg)

# outputs subreddits sorted by toxicity percentage
analysis = open('analysis.txt', 'w', encoding = 'utf-8', errors='ignore')
sortedScores = sorted(toxicScores.items(), key=lambda x:x[1], reverse = True)
for subreddit in sortedScores:
    # outputs subreddit's % of toxici comments
    hold = list(subreddit)
    analysis.write(hold[0])
    analysis.write("\t")
    analysis.write(str(hold[1]))
    analysis.write("\t")

    # outputs subreddit/s identifier ratios
    holdStat = list(stats[hold[0]])
    for s in holdStat:
        analysis.write(str(s))
        analysis.write("\t")
    analysis.write("\n")