# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:23:07 2019

@author: Morten
"""
#Pakker
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import re

#Read csv
stock = 'tsla'
tsla30_red = tsla
#tweets_reduced = pd.read_csv('C://Users/Morten/Desktop/Speciale/Python/Stock data/'+str(stock)+'.csv', encoding = 'latin1',lineterminator='\n')
tsla30 = pd.read_csv('C://Users/Morten/Desktop/Speciale/Python/3 Text trimming/tsla30_red.csv', encoding = 'latin1',lineterminator='\n')
amat30 = pd.read_csv('C://Users/Morten/Desktop/Speciale/Python/3 Text trimming/amat30.csv', encoding = 'latin1',lineterminator='\n')
tsla30_red = pd.read_csv('C://Users/Morten/Desktop/Speciale/Python/tsla30nycsv.csv', encoding = 'latin1',lineterminator='\n')
amat30_red = pd.read_csv('C://Users/Morten/Desktop/Speciale/Python/3 Text trimming/amat30_red.csv', encoding = 'latin1',lineterminator='\n')


list(tsla30.columns.values) 
#change everything to lower case
tsla30['text'] = tsla30.text.map(lambda x: x.lower())
amat30['text'] = amat30.text.map(lambda x: x.lower())
tsla30_red['text'] = tsla30_red.text.map(lambda x: x.lower())
amat30_red['text'] = amat30_red.text.map(lambda x: x.lower())


#Remove emojis
tsla30['text'] = tsla30['text'].str.replace('\<(.*?)\>', '')
amat30['text'] = amat30['text'].str.replace('\<(.*?)\>', '')
tsla30_red['text'] = tsla30_red['text'].str.replace('\<(.*?)\>', '')
amat30_red['text'] = amat30_red['text'].str.replace('\<(.*?)\>', '')


#Remove line shifts (\r\n)
tsla30['text'] = tsla30['text'].str.replace('\r\n', '')
amat30['text'] = amat30['text'].str.replace('\r\n', '')
tsla30_red['text'] = tsla30_red['text'].str.replace('\r\n', '')
amat30_red['text'] = amat30_red['text'].str.replace('\r\n', '')

#Substitute in callouts and links #LINKS ER ALLEREDE FJERNET I CSV TO TAB
#tsla30['text'] = tsla30['text'].str.replace('@\S+', 'callout')
#amat30['text'] = amat30['text'].str.replace('@\S+', 'callout')
tsla30['text'] = tsla30['text'].str.replace('http\S+', 'link')
amat30['text'] = amat30['text'].str.replace('http\S+', 'link')
tsla30_red['text'] = tsla30_red['text'].str.replace('http\S+', 'link')
amat30_red['text'] = amat30_red['text'].str.replace('http\S+', 'link')

#tsla30.to_excel('C://Users/Morten/Desktop/Speciale/Python/2 Stock data/Pre-stock/tsla30.xlsx',sheet_name='Sheet1')
#amat30.to_excel('C://Users/Morten/Desktop/Speciale/Python/2 Stock data/Pre-stock/amat30.xlsx',sheet_name='Sheet1')


#Fjern tegn og tal


tsla30_red['text'] = tsla30_red['text'].str.replace('\â', '')
tsla30_red['text'] = tsla30_red['text'].str.replace('\_+', '_')
tsla30_red['text'] = tsla30_red['text'].str.replace('\.+', ' punktummmmmmmm ')
tsla30_red['text'] = tsla30_red['text'].str.replace('\?+', ' spmtegnnnn ')
tsla30_red['text'] = tsla30_red['text'].str.replace('\!+', ' udrabbbbbb ')

tsla30['text'] = tsla30['text'].str.replace('[^\w\s]', '') 
tsla30['text'] = tsla30['text'].str.replace('\d', '')

amat30['text'] = amat30['text'].str.replace('[^\w\s]', '')
amat30['text'] = amat30['text'].str.replace('\d', '')

tsla30_red['text'] = tsla30_red['text'].str.replace('[^\w\s]', '') 
tsla30_red['text'] = tsla30_red['text'].str.replace('\d', '')

amat30_red['text'] = amat30_red['text'].str.replace('[^\w\s]', '') 
amat30_red['text'] = amat30_red['text'].str.replace('\d', '')
#OBS SE PÅ NUVÆRENDE TIDSPUNKT I KØRSEL. SE AMAT TWEET 1: LINK SMELTER SAMMEN MED ANDRE ORD UDEN MELLEMRUM

#Remove multiple whitespaces
tsla30['text'] = tsla30['text'].str.replace('\s\s+', ' ')
amat30['text'] = amat30['text'].str.replace('\s\s+', ' ')
tsla30_red['text'] = tsla30_red['text'].str.replace('\s\s+', ' ')
amat30_red['text'] = amat30_red['text'].str.replace('\s\s+', ' ')
#tweets_reduced['text'] = [re.sub('\s\s+', ' ', i) for i in tweets_reduced['text']]

#Remove leading and trailing whitespaces
tsla30['text'] = tsla30['text'].str.strip()
amat30['text'] = amat30['text'].str.strip()
tsla30_red['text'] = tsla30_red['text'].str.strip()
amat30_red['text'] = amat30_red['text'].str.strip()




amat30.to_csv('C://Users/Morten/Desktop/Speciale/Python/4 Model ready/amat30.csv', header=True, index=False, mode='w', quoting=None, quotechar='"', chunksize=None, tupleize_cols=None, encoding = 'latin1')
tsla30.to_csv('C://Users/Morten/Desktop/Speciale/Python/4 Model ready/tsla30.csv', header=True, index=False, mode='w', quoting=None, quotechar='"', chunksize=None, tupleize_cols=None, encoding = 'latin1')
amat30_red.to_csv('C://Users/Morten/Desktop/Speciale/Python/4 Model ready/amat30_red.csv', header=True, index=False, mode='w', quoting=None, quotechar='"', chunksize=None, tupleize_cols=None, encoding = 'latin1')
tsla30_red.to_csv('C://Users/Morten/Desktop/Speciale/Python/tsla30nycsv.csv', header=True, index=False, mode='w', quoting=None, quotechar='"', chunksize=None, tupleize_cols=None, encoding = 'latin1')



###########Slut udskriv til csv


#Tokenize
nltk.download()
tweets_reduced['text'] = tweets_reduced['text'].apply(nltk.word_tokenize)


#Stopwords     SKAL VI BRUGE DENNE?###########################################
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

tweets_reduced['text'] = [[word for word in text if not word in stop_words] for text in tweets_reduced['text']]


#Stemming
stemmer = PorterStemmer()
tweets_reduced['text'] = tweets_reduced['text'].apply(lambda x: [stemmer.stem(y) for y in x]) 
#tweets_reduced['text'] = [str(i) for i in tweets_reduced['text']]

tsla30_red = tsla30_red[tsla30_red.move_lag != 0]

################################### herfra hvordan prepper vi så for RNN og ikke Naive Bayes

#Transform model into occurences
tweets_reduced['text'] = tweets_reduced['text'].apply(lambda x: ' '.join(x))

tweets_reduced.to_csv('C://Users/crefsgaard/Documents/Stuff to keep/9. semester/Big Data Analytics/Projekt/toWordcloud.csv', index=False)

count_vect = CountVectorizer()  
counts = count_vect.fit_transform(tweets_reduced['text'])


#instead of word count per message, use term frequency inverse document frequency (see link to tutorial at the bottom)
transformer = TfidfTransformer().fit(counts)
counts = transformer.transform(counts) 


pos_count, neg_count = 0, 0
for num in tsla['movement']: 
      
    # checking condition 
    if num >= 0: 
        pos_count += 1
  
    else: 
        neg_count += 1
pos_count/(neg_count+pos_count)*100

pos_count, neg_count = 0, 0
for num in tsla['move_lag']: 
      
    # checking condition 
    if num >= 0: 
        pos_count += 1
  
    else: 
        neg_count += 1
pos_count/(neg_count+pos_count)*100

print("hej, mit navn er pik")



import numpy as np
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
import seaborn as sns

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.show() 


sns.distplot(tsla['movement'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

plt.hist(tsla['movement'], color = 'blue', edgecolor = 'black',
         bins = int(180/5))

    sns.distplot(tsla['movement'], hist = False, kde = True,ax=ax,
                 kde_kws = {'linewidth': 3})
ax = plt.subplots() 
