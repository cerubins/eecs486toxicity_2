#!/usr/bin/python3
'''
reddit_process_simple.py

Written by Kathleen Stone for EECS 486

General design by Alexandra Balahur in
Sentiment Analysis in Social Media Texts
https://aclanthology.org/W13-1617.pdf

How to run:
% python3 reddit_process_simple.py <directory_name>/

Where <directory_name> is the name of the directory containing .txt files to be processed line by line.

For example: 
% python3 reddit_process_simple.py comments/

General Structure:
This script opens each file in the directory, then for each line, replaces each word with its sentiment.
The word replacement follows these steps:
1) Lowercase all text
2) Remove all content in square brackets
3) Remove emojis
4) Remove links and attached text
5) Replace multi-punctuation with specific words
    - multiple periods in a row (eg. '..') replaced with "multistop"
    - multiple exclamation marks in a row replaced with "multiexclamation"
    - multiple question marks in a row replaced with "multiquestion"
6) emoticons replaced with words "positive", "negative", or "neutral"
    - based on emoticon sentiment list from SentiStrength dataset
7) slang is normalized, which allows step (10) to parse slang sentiment properly
    - same slang list as the one in Sentiment Analysis in Social Media Texts
    - https://slang.net/terms/social_media
8) text and punctuation are separated with whitespace
9) modifier words are replaced with corresponding: "diminisher", "intensifier", and "negator"
    - this indicates to BERT that the word following it is diminished, intensified, or negated by the word
    - modifier lists from The Impact of Intensifiers, Diminishers and Negations on Emotion Expressions
    -  https://www.semanticscholar.org/paper/The-Impact-of-Intensifiers-%2C-Diminishers-and-on-Strohm/d40e7b7df41a420e1bd456b39cae68726e3a0acb
10) English words (dicitionary from NLTK's English vocabulary list) are replaced by a sentiment from SentiWordNet
    - if positive = negative, word becomes "neutral"
    - if positive > negative and positive > 0.5, word becomes "hpositive"
    - if positive > negative, word becomes "positive"
    - if negative > positive and negative > 0.5, word becomes "hnegative"
    - if negative > positive, word becomes "negative"
The script writes the output files to a new output folder, titled <directory_name>_processed/
All files have "_processed" appended to their filenames.
'''
import sys
import os
import csv
import emoji
import re
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer

def main():
    # open folder containing data collection
    dir_name = sys.argv[1]
    # make new folder for processed files
    out_dir = dir_name.rstrip('/') + '_processed/'
    os.makedirs(out_dir)

    # open each file and process line by line
    for file_name in os.listdir(dir_name):
        with open(os.path.join(dir_name, file_name)) as f:
            out_file = str(file_name.rstrip('.txt')) + '_processed.txt'
            fout = open(os.path.join(out_dir, out_file), 'w')
            print('file ' + str(file_name) + ' opened')
            for line in f: 
                line = process(line)
                fout.write(str(line) + '\n')
            fout.close()
        print('file ' + str(file_name) + ' processed')

# input: string, the preprocessed text
# output: string, the processed text
# process processes the line of text through each step of the sentiment replacement process
def process(line):
    # lowercase everything
    line = line.lower()
    line = remove_meta(line)
    line = replace_meta(line)
    tokens = case_tokenize(line)
    tokens = replace_modifiers(tokens)
    line = " ".join(str(word) for word in tokens)
    return line


# input: string, unprocessed text
# output: string, without metatext or emojis
def remove_meta(line):
    # no_meta is the new line, which is appened char-by-char
    no_meta = ''
    in_brackets = False
    end_bracket = False
    for char in line:
        # if we are in brackets, don't append to new line
        if char == '[':
            in_brackets = True
        elif char == ']':
            end_bracket = True
        
        # if it is an emoji, don't append to new line
        if not in_brackets and not emoji.is_emoji(char):
            no_meta += char
        
        if end_bracket == True:
            in_brackets = False

    return no_meta


# input: string
# output: string wihtout any links, repeated punct, or emoticons
def replace_meta(line):
    # remove any characters attached to links (eg. parentheses)
    line = re.sub(r'\S*http\S+', '', line)
    
    # . repeated is replaced with "multistop"
    # ! repeated is replaced with "multiexclamation"
    # ? repaated is replaced with "multiquestion"\
    rule_stop = "[.]{2,}"
    rule_excl = "[!]{2,}"
    rule_quest = "[?]{2,}"
    line = re.sub(rule_stop, ' multistop ', line)
    line = re.sub(rule_excl, ' multiexclamation ', line)
    line = re.sub(rule_quest, ' multiquestion ', line)

    # positve emoticons replaced with "positive"
    # negative emotioncs replaced with "negative"
    # neutral emoticons removed from the processed text
    for emote in emoticons:
        if emoticons[emote] == "neutral":
            line = line.replace(emote, "")
        else:
            line = line.replace(emote, emoticons[emote])
    
    return line


# input: string
# output: list of normalized, separated tokens
def case_tokenize(line):
    # replace weird quotations with python quotations
    line = line.replace('’', "'")
    # replace slang
    for key in slang:
        line = line.replace(' ' + str(key) + ' ', ' ' + str(slang[key]) + ' ')

    # split by whitespace
    split_text = line.split()

    period_list = []
    # separate . with <anychar>.<whitespace>
    # exceptions include:
        # acronyms: .<at least 1 alpha>.<whitespace>
        # abbreviations: .<at least 1 alpha>
        # numbers: .<at least 1 numb>
    per_accro = "[.][a-zA-Z]+[.]$"
    per_abbrev = "[.][a-zA-Z]+"
    per_number = "[.][0-9]+"
    for token in split_text:
        if not re.search("[0-9]+", token): 
            if token == '.':
                period_list.append(token)
            elif "." in token and not re.search(per_abbrev, token) and not re.search(per_number, token) and not re.search(per_accro, token):
                split = token.split('.')
                i = 0
                for item in split:
                    if item == '':
                        period_list.append('.')
                    else:
                        period_list.append(item)
                        if len(split) > i + 1 and split[i+1] != '':
                            period_list.append('.')
                    i += 1
            else:
                period_list.append(token)

    comma_list = []
    # separate ,
    # exceptions include
    # numbers: <num>,<num>
    comma_num = "[0-9][,][0-9]"
    for token in period_list:
        if "," in token and not re.search(comma_num, token):
            split = token.split(',')
            i = 0
            for item in split:
                if item == '':
                    comma_list.append(',')
                else:
                    comma_list.append(item)
                    if len(split) > i + 1 and split[i+1] != '':
                        comma_list.append(',')
                i += 1
        else:
            comma_list.append(token)

    slash_list = []
    # separate /
    # exceptions include
    # dates: <num>/<num>
    slash_date = "[0-9][/][0-9]"
    slash_user = "[uU][/]"
    slash_subreddit = "[rR][/]"
    for token in comma_list:
        if "/" in token and not re.search(slash_date, token) and not re.search(slash_user, token) and not re.search(slash_subreddit, token):
            split = token.split('/')
            i = 0
            for item in split:
                if item == '':
                    slash_list.append('/')
                else:
                    slash_list.append(item)
                    if len(split) > i + 1 and split[i+1] != '':
                        slash_list.append('/')
                i += 1
        else:
            slash_list.append(token)
    
    dash_list = []
    # separate -
    # exceptions include
    # words: <alpha>-<alpha>
    dash_word = "[a-zA-Z][-][a-zA-Z]"
    for token in slash_list:
        if token == "-":
            dash_list.append(token)
        elif "-" in token and not re.search(dash_word, token):
            split = token.split('-')
            i = 0
            for item in split:
                if item == '':
                    dash_list.append('-')
                else:
                    period_list.append(item)
                    if len(split) > i + 1 and split[i+1] != '':
                        dash_list.append('-')
                i += 1
        else:
            dash_list.append(token)
    
    contract_list = []
    # separate contractions
    # parse list and replace contractions with expanded word(s)
    for token in dash_list:
        if token in contractions:
            temp_contracts = contractions[token].split()
            for contract in temp_contracts:
                contract_list.append(contract)
        else:
            contract_list.append(token)

    possess_list = []
    # separate 's<whitespace>
    # <whitespace makes sure quotes aren't split strangely
    # for example, if a quote starts with word beginning with "s", we don't want it to count as a possessive
    possess_str = "['][s]$"
    for token in contract_list:
        if re.search(possess_str, token):
            split = token.split("'")
            possess_list.append(split[0])
            possess_list.append("'s")
        else:
            possess_list.append(token)
    # replace contractions with de-punctuated versions

    final_list = []
    # separate misc. punctuation
    # ? ! : ; [ ] { } ( ) "
    # each becomes an individual token
    misc_punct = ["?", "!", ":", ";", "[", "]", "{", "}", "(", ")", '"', '“', '”']
    for token in possess_list:
        curr_str = ""
        for char in token:
            if char in misc_punct:
                if curr_str != "":
                    final_list.append(curr_str)
                final_list.append(char)
                curr_str = ""
            else:
                curr_str += char
        if curr_str != "":
            final_list.append(curr_str)

    return final_list


# input: tokens
# output: tokens, modifiers replaced properly
def replace_modifiers(tokens):
    new_tokens = []
    appended = False
    for token in tokens:
        # replace diminishers
        for word in diminishers:
            if token == word:
                new_tokens.append("diminisher")
                appended = True
        # replace intensifiers
        for word in intensifiers:
            if token == word:
                new_tokens.append("intensifier")
                appended = True
        # replace negators
        for word in negators:
            if token == word:
                new_tokens.append("negator")
                appended = True
        # if not replaced, check if in dictionary
        if not appended:
            new_tokens.append(word_affect(token))
        appended = False

    return new_tokens


# input: string, word
# output: string, word affect or just word
def word_affect(word):
    if word in english_vocab:
        # check if word is in synsets list after parsing to singular
        if list(swn.senti_synsets(word)) == []:
            if word == porter.stem(word):
                return word
            return word_affect(porter.stem(word))
        
        type = ''
        pos_score = list(swn.senti_synsets(word))[0].pos_score()
        neg_score = list(swn.senti_synsets(word))[0].neg_score()
        #obj_score = list(swn.senti_synsets(word))[0].obj_score()

        if pos_score > neg_score and pos_score >= 0.5:
            type = "hpositive"
            #if obj_score > pos_score:
            #    type = "neutral"
        elif pos_score > neg_score:
            type = "positive"
            #if obj_score > pos_score:
            #    type = "neutral"
        elif neg_score > pos_score and neg_score >= 0.5: 
            type = "hnegative"
            #if obj_score > neg_score:
            #    type = "neutral"
        elif neg_score > pos_score:
            type = "negative"
            #if obj_score > neg_score:
            #    type = "neutral"
        else:
            type = "neutral"
        
        return type
    return word


# global variables
if __name__ == "__main__":
    # make emoticon dictionary
    # emoticons[<emoticon>] = "positive" | "negative" | "neutral"
    emoticons = {}
    with open('process_files/emoticons.csv', newline='', encoding='iso-8859-1') as f:
        csvread = csv.reader(f, delimiter=',')
        for row in csvread:
            value = ""
            if row[1] == '-1':
                value = "negative"
            elif row[1] == '0':
                value = "neutral"
            elif row[1] == '1':
                value = "positive"
            emoticons[row[0]] = value
    
    # make intensifer list
    intensifiers = []
    with open('process_files/intensifiers.csv', encoding='utf-8') as f:
        csvread = csv.reader(f, delimiter=',')
        for row in csvread:
            intensifiers.append(row[0])
    
    # make diminisher list
    diminishers = []
    with open('process_files/diminishers.csv', encoding='utf-8') as f:
        csvread = csv.reader(f, delimiter=',')
        for row in csvread:
            diminishers.append(row[0])

    # make negator list
    negators = []
    with open('process_files/negators.csv', encoding='utf-8') as f:
        csvread = csv.reader(f, delimiter=',')
        for row in csvread:
            negators.append(row[0])

    # make contractions list
    contractions = {}
    with open('process_files/contractions.csv', encoding='utf-8') as f:
        csvread = csv.reader(f, delimiter=',')
        for row in csvread:
            contractions[row[0]] = row[1]

    # make slang list
    slang = {}
    with open('process_files/slang.tsv', encoding='utf-8') as f:
        csvread = csv.reader(f, delimiter='\t')
        for row in csvread:
            slang[row[0]] = row[1]

    # English vocabulary list
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    porter = PorterStemmer()
    
    main()