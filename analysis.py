import glob

# iterates through all subreddit BERT predictions
toxicScores = dict()
for file in glob.glob('output/*'):
    with open(file) as f:
        subreddit = file.lstrip('output/')
        lines = f.readlines()

        # counts number of "toxic" labeled comments in identifier
        lineCount = 0
        toxicCount = 0
        for line in lines:
            lineCount += 1
            if lineCount % 2 == 0 and 'Non' not in line:
                toxicCount += 1

        toxicScores[subreddit] = toxicCount/len(lines)

# outputs subreddits sorted by toxicity percentage
sortedScores = sorted(toxicScores.items(), key=lambda x:x[1], reverse = True)
print(sortedScores)
