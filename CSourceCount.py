# import required library functions
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
# special library for Word Clouds
from wordcloud import WordCloud, STOPWORDS
import glob

dirFiles = glob.glob("/home/praveen/Perforce/irbmlz_p2_canda_CAN_6684/**/*.c", recursive = True)

fileBuffer = []

for eachFile in dirFiles:
    with open(eachFile, encoding="utf-8", errors='ignore') as srcFile:
        fileBuffer.append((srcFile.read().split()))
# load list of common words to remove from consideration
stopwords = set(STOPWORDS)

# open the file and read it into a variable


# instantiate a word cloud object
novel_wc = WordCloud(
    background_color='white',
    max_words=20000,
    stopwords=stopwords
)

fileString = " ".join(str(x) for x in fileBuffer)

# generate the word cloud data
novel_wc.generate(fileString)

# display the word cloud
plt.imshow(novel_wc, interpolation='bilinear')
plt.axis('off')
plt.show()

print('Done')