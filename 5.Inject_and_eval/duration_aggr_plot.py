#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 19:20:10 2019

@author: felipeadachi
"""

import pandas as pd
import matplotlib.pyplot as plt
df_X=pd.read_csv('byduration_GX.csv')
df_Y=pd.read_csv('byduration_GY.csv')
df_Z=pd.read_csv('byduration_GZ.csv')

print(df_X)
print(df_Y)
print(df_Z)

df = pd.DataFrame(index=[10,25,50,100])
#Nota: como na dissertação a notação está trocada, inverte-se GYROX com GYROY na hora de plotar
df['GYROX']=df_Y['f_score'].values
df['GYROY']=df_X['f_score'].values
df['GYROZ']=df_Z['f_score'].values

#df.pivot(index='channel', columns='ab', values='booked').plot(kind='bar')
print(df)
df.plot(kind='bar')
plt.xticks(rotation='horizontal')
plt.show()