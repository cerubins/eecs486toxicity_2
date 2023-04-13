import glob
from pathlib import Path
import os
import emoji
import re

# stitches together the split up prediction files
# subreddits = ["AnimalCrossing","baseball","bettercallsaul","Boxing","classicalmusic",
# "dc_cinematic","DestinyTheGame","dundermifflin","eldenring","electronicmusic","ffxiv",
# "formula1","FortNiteBR","Genshin_Impact","hiphopheads","hockey","houseofthedragon","indieheads",
# "kanye","kpop","kuwtk","leagueoflegends","LofiHipHop","lostarkgame","marvelstudios","metal",
# "minecraft","mma","nba","nfl","popheads","PremierLeague","Rap","rpclipsgta","rupaulsdragrace",
# "soccer","squaredcircle","starwars","strangerthings","thebachelor"]

# def tryint(s):
#     try:
#         return int(s)
#     except:
#         return s

# def alphanum_keys(s):
#     return [tryint(c) for c in re.split('([0-9]+)', s)]

# data = Path('output/')
# oldpwd = os.getcwd()
# for folder in data.iterdir():
#     subreddit = folder.name
    
#     comments = []
#     # newData = Path(folder)
#     # list(newData.iterdir()).sort(key=alphanum_keys)
#     # print('output/' + str(folder.name) + '/*')
#     li = glob.glob('output/' + str(folder.name) + '/*')
#     li.sort(key=alphanum_keys)
#     # print(li)
#     for file in li:
#         print(file)
#         with open(file, 'r', encoding = 'utf-8', errors='ignore') as f:
#             comment = f.readlines()
#             comments.append(comment)

#     output = open('output_stitched/' + subreddit, 'w', encoding = 'utf-8', errors='ignore')
#     for file in comments:
#         output.writelines(file)



# iterates through all subreddit BERT predictions
toxicScores = dict()
for file in glob.glob('output_stitched/*'):
    with open(file, encoding = 'utf-8') as f:
        subreddit = file.removeprefix('output_stitched\\')
        lines = f.readlines()

        # finds original comments, before preprocessing
        unprocessed = open('comments_test/' + subreddit + '.txt', encoding = 'utf-8')
        unprocessLines = unprocessed.readlines()

        # counts number of "toxic" labeled comments in identifier
        lineCount = 0
        toxicCount = 0
        wordCount = 0
        index = 0
        pair = ("", "", "")
        comments = []

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
            else:
                update = list(pair)
                update[2] = line
                pair = tuple(update)
                comments.append(pair)

            lineCount += 1
            wordCount += len(line.split())

        toxicScores[subreddit] = toxicCount/len(lines)

# outputs subreddits sorted by toxicity percentage
sortedScores = sorted(toxicScores.items(), key=lambda x:x[1], reverse = True)
print(sortedScores)
print(comments)
print(subreddit)