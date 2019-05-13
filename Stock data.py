# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:35:43 2019

@author: Morten
"""

#stock movement er tilføjet i excel
#tidszoner er tilpasset, så tweets kan linkes med kurser. API var GMT, Bloomberg var GMT+1, derfor har tweets fået +1time
#volume, 10- og 30min bevægelser er tilføjet.
#dato-delen er fjernet fra start tidspunkter, så vi kun har tid.

import pandas as pd

#Indlæs post-stock data
tsla = pd.read_excel('C:/Desktop/Python/tsssla.xlsx', encoding = 'latin1',lineterminator='\n')
#patv = pd.read_csv('C://Users/Morten/Desktop/Speciale/Python/2 Stock data/Post-stock/values/patv.csv', encoding = 'latin1',lineterminator='\n')
#amd = pd.read_excel('C://Users/Morten/Desktop/Speciale/Python/2 Stock data/Post-stock/values/amd.xlsx', encoding = 'latin1',lineterminator='\n')
amat = pd.read_excel('C://Users/Morten/Desktop/Speciale/Python/tsla30ny.xlsx', encoding = 'latin1',lineterminator='\n')
#johnson = pd.read_csv('C://Users/Morten/Desktop/Speciale/Python/2 Stock data/Post-stock/values/johnson.csv', encoding = 'latin1',lineterminator='\n')

#Remove tweets outside exhange-opening hours (1530-2200 minus saturday,sunday)

#30 min interval
list(tsla.columns.values)
amat.index = pd.to_datetime(amat['start30'])
amat = amat.between_time('15:00:00' , '21:30:00')
amat.index = pd.to_datetime(amat['timestamp'])
amat = amat[amat.index.dayofweek < 5]


tsla = amat
tsla = tsla.drop(['Unnamed: 13',
 'Unnamed: 14',
 'Unnamed: 15',
 'Unnamed: 16',
 'Unnamed: 17',
 'Unnamed: 18',
 'Unnamed: 19',
 'Unnamed: 20',
 'Unnamed: 21',
 'Unnamed: 22',
 'Unnamed: 23',
 'Unnamed: 24',
 'Unnamed: 25',
 'Unnamed: 26',
 'Unnamed: 27'],axis=1)


tsla.index = pd.to_datetime(tsla['timestamp'])
tsla = tsla.between_time('15:57:31' , '21:32:29')
tsla.index = pd.to_datetime(tsla['timestamp'])
tsla = tsla[tsla.index.dayofweek < 5]


amat_red = amat.between_time('16:00:00' , '21:30:00')
tsla_red = tsla.between_time('16:00:00' , '21:30:00')
#Write to csv in order to process text and excess variables in next window
amat.to_csv('C://Users/Morten/Desktop/Speciale/Python/3 Text trimming/amat30.csv', header=True, index=False, mode='w', quoting=None, quotechar='"', chunksize=None, tupleize_cols=None, encoding = 'latin1')
amat_red.to_csv('C://Users/Morten/Desktop/Speciale/Python/3 Text trimming/amat30_red.csv', header=True, index=False, mode='w', quoting=None, quotechar='"', chunksize=None, tupleize_cols=None, encoding = 'latin1')
tsla.to_csv('C://Users/Morten/Desktop/Speciale/Python/Nye intervaller/Kun børs tweets/tsla30_red.csv', header=True, index=False, mode='w', quoting=None, quotechar='"', chunksize=None, tupleize_cols=None, encoding = 'latin1')
tsla_red.to_csv('C://Users/Morten/Desktop/Speciale/Python/3 Text trimming/tsla30_red.csv', header=True, index=False, mode='w', quoting=None, quotechar='"', chunksize=None, tupleize_cols=None, encoding = 'latin1')


#10 min interval
tsla.index = pd.to_datetime(tsla['start10min'])
tsla = tsla.between_time('15:30:00' , '21:50:00')
tsla.index = pd.to_datetime(tsla['timestamp'])
tsla = tsla[tsla.index.dayofweek < 5]

#Write to csv in order to process text and excess variables in next window
amat.to_csv('C://Users/Morten/Desktop/Speciale/Python/3 Text trimming/amat10.csv', header=True, index=False, mode='w', quoting=None, quotechar='"', chunksize=None, tupleize_cols=None, encoding = 'latin1')
tsla.to_csv('C://Users/Morten/Desktop/Speciale/Python/3 Text trimming/tsla10.csv', header=True, index=False, mode='w', quoting=None, quotechar='"', chunksize=None, tupleize_cols=None, encoding = 'latin1')






#Optional
#Add categorical movement class e.g (1-5) with sd.dev determining movement intervals within classes
