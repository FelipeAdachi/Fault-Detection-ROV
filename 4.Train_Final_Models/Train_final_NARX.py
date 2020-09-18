#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:10:05 2019

@author: felipeadachi
"""

path='/home/felipeadachi/Dados/Kfolds_cpu'
train_file='/Whole_Train.json'
test_file='/Whole_Test.json'

import json
import os
import collections
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.callbacks import CSVLogger
import pickle

def main():
    df_train=pd.DataFrame()
    df_val=pd.DataFrame()
    
    aux=[]
    
    js_train=read_json_content(path+train_file)
    js_val=read_json_content(path+test_file)

    #---------Gera os atributos passados de acordo com which_topast    
    
    for att in which_topast:
        js_train=past_commands(js_train,att)
        js_val=past_commands(js_val,att)
        att2=att[0]+'_t'+str(att[1])
        aux.append(att2)
    
    which_tokeep=['GYROZ']
    which_tokeep=which_tokeep+aux
    #print(which_tokeep)

    #---------Gera os dataframes a partir dos json    
    
    df_train=df_train.append(pd.DataFrame(js_train))
    df_val=df_val.append(pd.DataFrame(js_val))
    
    #---------Mantem no dataframe apenas os atributos alvo e independentes
    
    for col in df_train.columns:
            if col not in which_tokeep:
                df_train=df_train.drop(columns=col)
    for col in df_val.columns:
        if col not in which_tokeep:
            df_val=df_val.drop(columns=col)

    with pd.option_context('display.max_columns',None):
        print(df_train.head())
    
    df_train = df_train.apply(pd.to_numeric, errors='coerce')
    df_train=df_train.dropna()
    df_val = df_val.apply(pd.to_numeric, errors='coerce')
    df_val=df_val.dropna()
    
    print(len(df_train))
    print(len(df_val))
    
    #---------gera numpy para x e y, para train e validation
    
    
    y_train=df_train['GYROZ'].values
    y_val=df_val['GYROZ'].values
    
    #print(y.shape)
    
    y_train=y_train.reshape(-1,1)
    y_val=y_val.reshape(-1,1)
    
    scaler_y_train = preprocessing.MinMaxScaler()
    scaler_y_val = preprocessing.MinMaxScaler()
    
    
    scaler_y_train.fit(y_train)
    
    
    
    file = open('rf_y.pickle', 'wb')
    pickle.dump(scaler_y_train,file)
    file.close()
    
    y_train = scaler_y_train.transform(y_train)
    
    y_val = scaler_y_val.fit_transform(y_val)
    
    y_train=y_train.ravel()
    y_val=y_val.ravel()
    print(y_train)
    print(y_val)
    
    
    x_train=df_train.drop(columns='GYROZ').values
    x_val=df_val.drop(columns='GYROZ').values
    
    scaler_x_train = preprocessing.MinMaxScaler()
    scaler_x_val = preprocessing.MinMaxScaler()
    
    scaler_x_train.fit(x_train)
    
    #scalerx=min_max_scaler_x.fit(x_train)

    
    file = open('rf_x.pickle', 'wb')
    pickle.dump(scaler_x_train,file)
    file.close()

    x_train = scaler_x_train.transform(x_train)
    x_val = scaler_x_val.fit_transform(x_val)
    
    print(x_train.shape)
    print(x_val.shape)
    
    
    model = Sequential()
    model.add(Dense(29, input_dim=29, kernel_initializer='uniform', activation='tanh'))
    model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1))    

    # Compile model
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    opt=SGD(lr=0.1, momentum=0.9)

    # Compile model
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    opt=SGD(lr=0.2, momentum=0.9)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])

    csv_logger = CSVLogger('rf.csv', append=True, separator=';')
    model.fit(x_train, y_train,callbacks=[csv_logger], epochs=100, batch_size=200,validation_data=(x_val,y_val), verbose=1)
    mj = model.to_json()
    jf = open("rf.json","w")
    jf.write(mj)
    jf.close()
    model.save_weights("rf.h5")    

    print(model.evaluate(x_val,y_val))
# =============================================================================
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Test'], loc='upper left')
#     plt.show()
# 
# =============================================================================
    
#---------------------#---------------------#---------------------#---------------------#---------------------#
def past_commands(content,att):
    
    data_list=content
    i=0
    
        
    while (i<len(data_list)):
        
        #get_token retorna o número de vezes permitido para pegar os comandos passados, com base nos flags "has_elapsed" de cada timestep
        token=get_token(i,data_list,att)
        #print("token for timestep:",i,"is:",token)
        data=data_list[i]
        tkn={}
        
        tkn[att[0]]=token
        
        
        if i<att[1]:
            data[att[0]+'_t'+str(att[1])]=np.nan
        else:
            try:
                if token>=att[1]:
                    data[att[0]+'_t'+str(att[1])]=data_list[i-att[1]][att[0]]
                else:
                    data[att[0]+'_t'+str(att[1])]=np.nan
            except KeyError:
                data[att[0]+'_t'+str(att[1])]=np.nan                    
        i+=1
    return content
#---------------------#---------------------#---------------------#---------------------#---------------------#
def get_token(i,data_list,att):
    j=0
    token=0
    while j<att[1]:
        if data_list[i-j]['has_elapsed'] == 0:
            token+=1
        else:
            return token
        
        j+=1
    return token
#---------------------#---------------------#---------------------#---------------------#---------------------#
def read_json_content(filename):
    with open(filename, 'r') as f:
        return json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(f.read(os.stat(filename).st_size))
#---------------------#---------------------#---------------------#---------------------#---------------------#


which_topast = [
    ["mtarg1",1],
    ["mtarg3",1],
    
    ["mtarg2",1],
    ["mtarg2",2],
    

    ["mtarg1",2],
    ["mtarg3",2],

    ["roll",1],

    ["pitch",1],
    #["deapth",0],
    
    ["GYROX",1],
    ["GYROX",2],
    
    ["GYROY",1],
    
    ["SC1I",1],
    ["SC1I",2],
    
    
    ["SC3I",1],
    ["SC3I",2],
    
    
    ["BT1I",1],
    ["BT1I",2],
    
    ["BT2I",1],
    ["BT2I",2],
    
    
    ["vout",1],
    
    ["iout",1],
    
    ["cpuUsage",1],
    
    ["GYROZ",1],
    ["GYROZ",2],
    ["GYROZ",3],
    ["GYROZ",4],
    ["GYROZ",5],
    ["GYROZ",6],
    ["GYROZ",7]
    
    #["temp",0]
    
    ]


if (__name__=='__main__'):
    main();
