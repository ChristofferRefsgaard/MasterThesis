
import numpy as np
import tensorflow as tf
import random as rn

import os
import pandas as pd

from keras.callbacks import ModelCheckpoint

from keras import backend as K
import keras
from keras.layers.normalization import BatchNormalization
#from keras_layer_normalization import LayerNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, concatenate
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from keras import initializers, regularizers, constraints, optimizers, layers

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from nltk.corpus import stopwords

import time


embed_size = 30 #embedding vektor længde
max_features = 25000 #antal unikke ord (antal søjler i embedding matricen)
maxlen = 86

###Train your own word embeddings###
print('Loading data...')
tsla30 = pd.read_csv('C://Users/crefsgaard/Desktop/speciale/data/tsla30nycsv.csv', encoding = 'latin1',lineterminator='\n')


###PRE-PROCESS TEXT DATA###
tokenizer = TreebankWordTokenizer()
tsla30['text'] = [tokenizer.tokenize(tweet) for tweet in tsla30['text']]

#Find længste tweet og sæt maxlen til dette
maxlen = 0
for i in range(len(tsla30['text'])):
    a = len(tsla30['text'][i])
    if a > maxlen:
        maxlen = a
maxlen = maxlen+2 #max antal ord i et tweet

#Stemming
stemmer = PorterStemmer()
tsla30['text'] = tsla30['text'].apply(lambda x: [stemmer.stem(y) for y in x])

#Calculating unique occurences
DF = {}
for tweet in range(len(tsla30['text'])):
    tokens = tsla30['text'][tweet]
    for w in tokens:
        try:
            DF[w].add(tweet)
        except:
            DF[w] = {tweet}

#Antal ord i ordbogen
for i in DF:
    DF[i] = len(DF[i])
total_vocab = [x for x in DF]

#Calculating frequencies
DF2 = DF.copy()
for w in DF2:
    DF2[w] = DF2[w]/len(tsla30)


#sorted_dict = sorted(DF2.items(), key=lambda kv: kv[1])


#Low frequent stopwords
lowfreq_stopWords = []
for word in DF:
    if DF[word] < 5:
        lowfreq_stopWords.append(word)


test = lowfreq_stopWords[40000:45000]

print('removing stopwords')
start = time.time()
tsla30['text'] = [[word for word in tweet if not word in test] for tweet in tsla30['text']]
end = time.time()
print(end-start)
###FINISHED###


###Ord totaler###

#total_vocab: 58909
#lowfreq length: 40939
#total vocab after removing freq words: 17.970
#Når vi detokenizer -> fil -> indlæs -> tokenize går vi fra 17.970 til 18.057

#Når vi bare kører fjern freq words, detokenizer, og tokenizer i keras, så er der 17997 unikke ord.

#corpus specific: 3
#nltk stopwords: 115
#total: 118

#Når alle stopord er sorteret fra er der: 17947 unikke ord. Der burde have været 18.057-118=17939, men der er nogle få stopord som ikke er i vores tweets.


#Skriv teksterne, renset for stopord, til fil
tsla_TEST = tsla30.copy()
#After removing non-frequent words, detokenize and write to .csv
detokenizer = TreebankWordDetokenizer()
tsla30['text'] = [detokenizer.detokenize(tweet) for tweet in tsla30['text']]
os.chdir('C://Users/crefsgaard/Desktop/speciale/data')
tsla30.to_csv('nonfreqRemoved.csv', encoding="latin1", index = False)




###Indlæs fil hvor non freq ord er fjernet, og fjern nltk stopord mv.###
tesla = pd.read_csv('C://Users/Morten/Desktop/Speciale/Python/nonfreqRemoved.csv', encoding = 'latin1',lineterminator='\n')


#Tokenize
tokenizer = TreebankWordTokenizer()
tesla['text'] = [tokenizer.tokenize(tweet) for tweet in tesla['text']]

nltk_stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'very', 'having', 'with', 'they', 'own', 'an', 
                  'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 
                  'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'me', 'were', 'her', 'more', 'himself', 
                  'this', 'should', 'our', 'their', 'while', 'both', 'to', 'ours', 'had', 'she', 'all', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 
                  'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'why', 'so', 'can', 'did', 'he', 'you', 'herself', 'has', 'just', 
                  'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 'being', 'if', 'theirs', 'my', 'a', 'by', 'doing', 'it', 'how', 
                  'further', 'was', 'here', 'than']        
#Stem stopwords
stemmer = PorterStemmer()
nltk_stopwords = [stemmer.stem(x) for x in nltk_stopwords]


#Calculating unique occurences
DF = {}
for tweet in range(len(tesla['text'])):
    tokens = tesla['text'][tweet]
    for w in tokens:
        try:
            DF[w].add(tweet)
        except:
            DF[w] = {tweet}

#Antal ord i ordbogen
for i in DF:
    DF[i] = len(DF[i])
total_vocab = [x for x in DF]

#Calculating frequencies
DF2 = DF.copy()
for w in DF2:
    DF2[w] = DF2[w]/len(tesla)



#All stopwords
all_stopWords = []
for word in DF2:
    if DF2[word] > 0.5:
        all_stopWords.append(word)

for word in nltk_stopwords:
    all_stopWords.append(word)


print('removing corpus specific and predefined stopwords')
start = time.time()
tesla['text'] = [[word for word in tweet if not word in all_stopWords] for tweet in tesla['text']]
end = time.time()
print(end-start)


#Beregn maxlen
maxlen = 0
for tweet in tesla['text']:
    b = len(tweet)
    if b > maxlen:
        maxlen = b
maxlen=maxlen+1
#############################

###Train Word2Vec model###
start = time.time()
w2v = Word2Vec(tesla['text'], size=embed_size, min_count=5, window=5, sg = 1, negative=15, iter=10)
len(list(w2v.wv.vocab)) #Antal ord der er lavet embeddings for
end = time.time()
print(end-start)

#Get trained embeddings and check similarities
word_vectors = w2v.wv
result = word_vectors.similar_by_word("market")
print("Most similar to 'model':\n", result[:3])

#Map words to indexes
word2id = {k: v.index for k, v in word_vectors.vocab.items()}

#Save trained embeddings
os.chdir('C://Users/Morten/Desktop/Speciale/Python')
word_vectors.save_word2vec_format('trainedEmb.txt', binary=False)
###FINISHED###




#Write tokenized sentences back to real sentences in order to number tokenize the sentences with keras later
detokenizer = TreebankWordDetokenizer()
tesla['text'] = [detokenizer.detokenize(tweet) for tweet in tesla['text']]


#Normalize input data and split to train/test
x = tesla.copy()
#x = x.sample(frac=1).reset_index(drop=True) #Så x er det samplede data
x=x.sort_values(by='timestamp.1')
y = x['Movement']
x = x[['text', 'followers_count', 'vol_lag', 'verified', 'move_lag\r\r']]
#x = x[['text', 'followers_count', 'vol_lag', 'move_lag\r\r']]
x['followers_count'] = (x['followers_count']-x['followers_count'].mean())/x['followers_count'].std() #Normalisering af data
x['vol_lag'] = (x['vol_lag']-x['vol_lag'].mean())/x['vol_lag'].std() #Normalisering af data
x['move_lag\r\r'] = (x['move_lag\r\r']-x['move_lag\r\r'].mean())/x['move_lag\r\r'].std() #Normalisering af data

timestamp =tesla.copy()
timestamp =timestamp.sort_values(by='timestamp.1')
timestamp = timestamp['timestamp.1']
timestamp = timestamp[194102:]
buffer, timestamp = train_test_split(timestamp, test_size=0.5, random_state=42)

list(x.columns.values)
x1 = x[:194102]
x2 = x[194102:]
y1 = y[:194102]
y2 = y[194102:]
x1 = x1.sample(frac=1,random_state=42)
y1 = y1.sample(frac=1,random_state=42)

x = x1.append(x2)
y = y1.append(y2)


#Tokenize words to numbers, apply padding and define the final dataset
print('Tokenize data...')
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x['text']))
list_tokenized = tokenizer.texts_to_sequences(x['text'])




x_lstm = pad_sequences(list_tokenized, maxlen=maxlen)
x_nn = x[['followers_count', 'vol_lag', 'verified', 'move_lag\r\r']]
x#_nn = x[['followers_count', 'vol_lag', 'move_lag\r\r']]

x_train_nn  = x_nn[:194102]
x_train_lstm= x_lstm[:194102]
y_train     = y[:194102]

x_test_lstm_shuf  = x_lstm[194102:]
x_test_shuf       = x_nn[194102:]
y_test_shuf       = y[194102:]

x_val_nn, x_test_nn, y_val, y_test = train_test_split(x_test_shuf, y_test_shuf, test_size=0.5, random_state=42)
x_val_lstm, x_test_lstm, y_val, y_test = train_test_split(x_test_lstm_shuf, y_test_shuf, test_size=0.5, random_state=42)

#x_train_nn = x_train_nn.append(x_val_nn) 
#x_train_lstm = np.append(x_train_lstm,x_val_lstm,axis=0)
#y_train = y_train.append(y_val)

#x_train_nn, x_test_nn, y_train, y_test = train_test_split(x_nn, y, test_size=0.1, random_state=42)
#x_train_lstm, x_test_lstm, y_train, y_test = train_test_split(x_lstm, y, test_size=0.1, random_state=42)
print(len(x_train_nn), 'train sequences')
print(len(x_test_nn), 'test sequences')



print('Load our own pre-trained word embeddings...')
EMBEDDING_FILE = 'C://Users/Morten/Desktop/Speciale/Python/trainedEmb.txt'


#Læs pre-trained word embeddings, og definér matrix med disse
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE, encoding="utf-8"))
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std

word_index = tokenizer.word_index #liste med unikke ord i data
nb_words = min(max_features, len(word_index)) #hvis antal unikke ord overskrider det max antal, vi ønsker, vælges det
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size)) #initialisér embedding matricen randomly
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i-1] = embedding_vector
    

#TODO
#Næste kørsel: NN med 3 lag - google lige hvad fornuftigt valg af neuroner kan være.
#Kig på BATCH og LAYER normalization
#Overvej hvilke arguments vi skal tilføje til LSTMen og NNet
#Find ud af hvordan vi kan plotte trænings loss og val loss


print('defining model')

###BiLSTM med NN ovenpå###
#http://digital-thinking.de/deep-learning-combining-numerical-and-text-features-in-deep-neural-networks/
#Definér loss-funktionen: https://codeburst.io/neural-networks-for-algorithmic-trading-volatility-forecasting-and-custom-loss-functions-c030e316ea7e
#Loss funktionen skal skrives med Keras backend, når den skal burges i keras/tensorflow netværkene senere.
def stock_loss(y_true, y_pred):
    loss=-y_pred*y_true/K.sum(K.abs(y_pred))
    return K.mean(loss, axis=-1)

#Tilføj række for bias i embedding matricen
newrow = np.array([np.zeros(embed_size)])
A = np.append(newrow, embedding_matrix, axis = 0)

model = Sequential()

nlp_input = Input(shape=(maxlen,), name = 'nlp_input')
meta_input = Input(shape=(4,), name = 'meta_input')
emb = Embedding(nb_words+1, embed_size, weights = [A])(nlp_input)
nlp_out = Bidirectional(LSTM(maxlen, 
                             activation = 'tanh',  
                             use_bias = True,
                             unit_forget_bias = True, 
                             kernel_initializer = keras.initializers.glorot_uniform(seed =62)))(emb)
nlp_out = Dense(1, activation = 'linear', kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=0.05,seed=32),bias_initializer='zeros')(nlp_out)

x = keras.layers.concatenate([nlp_out, meta_input])
#x = concatenate([nlp_out, meta_input])
x = BatchNormalization()(x)

x = Dense(4, activation='relu', kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=0.25,seed=32))(x)
x = BatchNormalization()(x)
x = Dense(3, activation='relu', kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=0.30,seed=32))(x)
x = BatchNormalization()(x)
#x = Dense(4, activation='relu', kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=0.25,seed=32), kernel_regularizer=regularizers.l2(0.005))(x)
#x = Dropout(0.25)(x)
#x = Dense(2, activation='relu', kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=0.23,seed=42))(x)
#x = BatchNormalization()(x)
#x = Dropout(0.25)(x)
#x = Dense(3, activation='relu', kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=0.25,seed=45), kernel_regularizer=regularizers.l2(0.001))(x)
#x = BatchNormalization()(x)
#x = Dropout(0.25)(x)
#x = Dense(3, activation='relu', kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=0.22,seed=44))(x)
#x = Dense(5, activation='relu', kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=0.3,seed=43))(x)
x = Dense(2, activation='relu', kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=0.30,seed=41))(x)
x = BatchNormalization()(x)
x = Dense(1, activation='linear', kernel_initializer = keras.initializers.RandomNormal(mean=0,stddev=0.30,seed=52))(x)
model_lstmnn = Model(inputs=[nlp_input, meta_input], outputs=[x])

    
print('compile and train model')
start = time.time()
model_lstmnn.compile(loss=stock_loss,optimizer= keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))
filepath="C://Users/Morten/Desktop/Speciale/Python/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model_lstmnn.fit(x=[x_train_lstm, x_train_nn], y=y_train, batch_size=1024,epochs=75,validation_data=([x_val_lstm, x_val_nn], y_val), shuffle=True,callbacks=callbacks_list)
end = time.time()
print((end-start)/3600) #timer



#ting at teste, hvis det ikke virker:
#flere std.devs
#dropout
#lambda
#arkitektur
#Hvad med embedding size f.eks 50? Tjek
#uden præ-trænede?


print('make predictions and calculate returns')
y_lstmnn_pred = model_lstmnn.predict([x_test_lstm, x_test_nn], batch_size=1024, verbose = 1)
#np.unique(y_lstmnn_pred)

#gem tidligere resultater som vektorer
y_true = np.array(y_test)
y_pred = y_lstmnn_pred

y_pred.min()
y_pred.max()
y_pred.mean()
len(np.unique(y_pred))

#Beregn afkast
a = np.multiply(y_pred.T, y_true).T
a.sum()
###FINISHED###






#call back model:
model = Sequential()

nlp_input = Input(shape=(maxlen,), name = 'nlp_input')
meta_input = Input(shape=(4,), name = 'meta_input')
emb = Embedding(nb_words+1, embed_size, weights = [A])(nlp_input)
nlp_out = Bidirectional(LSTM(maxlen, 
                             activation = 'tanh',  
                             use_bias = True,
                             unit_forget_bias = True, 
                             kernel_initializer = keras.initializers.glorot_uniform(seed =62)))(emb)
nlp_out = Dense(1, activation = 'linear', kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=0.05,seed=32),bias_initializer='zeros')(nlp_out)

x = keras.layers.concatenate([nlp_out, meta_input])
#x = concatenate([nlp_out, meta_input])
x = BatchNormalization()(x)

x = Dense(4, activation='relu', kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=0.25,seed=32))(x)
x = BatchNormalization()(x)
x = Dense(3, activation='relu', kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=0.30,seed=32))(x)
x = BatchNormalization()(x)
#x = Dense(4, activation='relu', kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=0.25,seed=32), kernel_regularizer=regularizers.l2(0.005))(x)
#x = Dropout(0.25)(x)
#x = Dense(2, activation='relu', kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=0.23,seed=42))(x)
#x = BatchNormalization()(x)
#x = Dropout(0.25)(x)
#x = Dense(3, activation='relu', kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=0.25,seed=45), kernel_regularizer=regularizers.l2(0.001))(x)
#x = BatchNormalization()(x)
#x = Dropout(0.25)(x)
#x = Dense(3, activation='relu', kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=0.22,seed=44))(x)
#x = Dense(5, activation='relu', kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=0.3,seed=43))(x)
x = Dense(2, activation='relu', kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=0.30,seed=41))(x)
x = BatchNormalization()(x)
x = Dense(1, activation='linear', kernel_initializer = keras.initializers.RandomNormal(mean=0,stddev=0.30,seed=52))(x)
model_lstmnn = Model(inputs=[nlp_input, meta_input], outputs=[x])



#SLUT KOPIERING HER

model_lstmnn.load_weights('C://Users/Morten/Desktop/Speciale/Python/weights.best.hdf5')

model_lstmnn.compile(loss=stock_loss,optimizer= keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))
y_lstmnn_pred = model_lstmnn.predict([x_test_lstm, x_test_nn], batch_size=1024, verbose = 1)

#gem tidligere resultater som vektorer
y_true = np.array(y_test)
y_pred = y_lstmnn_pred

y_pred.min()
y_pred.max()
y_pred.mean()
len(np.unique(y_pred))

#Beregn afkast
a = np.multiply(y_pred.T, y_true).T
a.sum()
###FINISHED###

np.sum(y_pred)
#Udskriv ytrue, ypred og timestamp til grafudvikling
#gem y-pred i dokument hvis fornuftigt resultat



########SANDBOX######
###Training Multilayer Neural Network###
model_nn = Sequential()
model_nn.add(Dense(8, input_dim=x_nn_train.shape[1], activation = 'relu'))
model_nn.add(Dense(5, activation = 'relu'))
model_nn.add(Dense(1, activation = None))

#Compile model
model_nn.compile(loss='mean_squared_error', optimizer = 'adam', metrics = ['mean_squared_error'])

#Fit model
model_nn.fit(x_nn_train, y_nn_train, batch_size=32, epochs = 10, validation_split=0.2)

#Predictions
y_nn_pred = model_nn.predict(x_nn_test, batch_size=32, verbose=1)

###FINISHED###


#LSTM uden pre-trained embeddings
inp = Input(shape=(maxlen,))
x = Embedding(nb_words+1, embed_size)(inp)
x = Bidirectional(LSTM(100, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(100, activation="tanh")(x)
x = Dropout(0.25)(x)
x = Dense(1, activation=None)(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

model.fit(x_tr, y_train, batch_size=32, epochs=1, validation_split=0.2)

#returnerer array af predictions
y_lstm_pred = model.predict([x_te], batch_size=32, verbose=1)

y_NN_input = y_lstm_pred
###FINISHED###


#LSTM med pre-trained embeddings
print('Bidirectional LSTM...')
inp = Input(shape=(maxlen,))
x = Embedding(nb_words, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(100, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(100, activation="relu")(x)
x = Dropout(0.25)(x)
x = Dense(1, activation=None)(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

model.fit(x_tr, y_train, batch_size=32, epochs=1)
###FINISHED###


######TSNE plot########
def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in w2v.wv.vocab:
        tokens.append(w2v[word])
        labels.append(word)
    
    tsne_w2v = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000, random_state=23)
    new_values = tsne_w2v.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

tsne_plot(w2v)
############################



#https://machinelearningmastery.com/check-point-deep-learning-models-keras/

