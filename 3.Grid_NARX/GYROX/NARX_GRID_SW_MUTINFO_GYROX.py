#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:20:17 2019

@author: felipeadachi
"""

from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
set_random_seed(2)

path='/home/felipeadachi/Dados/Kfolds_cpu/Fold_'
train_file='/Train_'
val_file='/Validate_'
filename='/home/felipeadachi/Dados/Kfolds_cpu/Models_GridSearch/GYROX/SW_MUTINFO/log_narx.txt'
path_models='/home/felipeadachi/Dados/Kfolds_cpu/Models_GridSearch/GYROX/SW_MUTINFO'

import json
import os
import collections
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from keras.optimizers import SGD
import pickle
import statistics as st
from itertools import product
from keras.callbacks import CSVLogger
from collections import OrderedDict


db_json='/home/felipeadachi/Dados'


def main():

    for k in range(1,6):
        which='which_topast_fold_'+str(k)
        print(len(eval(which)))
    

    df=pd.DataFrame()
    with open(db_json+'/DB-Train','rb') as f:
        json_content=pickle.load(f)
    
    df=pd.DataFrame(json_content)
    print("len1>",len(df))
    kf = KFold(n_splits=5,shuffle=False,random_state=2)
    
    
    for col in df.columns:
        if col not in which_tokeep_0:
            df=df.drop(columns=col)
    
    df = df.apply(pd.to_numeric, errors='coerce')
    
    try:
        df=df.drop(df['LACCX'].idxmax())
        df=df.drop(df['LACCX'].idxmin())

    
    
    except KeyError:
        pass
    
    #with pd.option_context('display.max_columns',None):
    #    print("df head>",df.head())
    print("len 2>",len(df))
    df = df[df.cpuUsage != 1]
    print("len 3>",len(df))
    
    df = df[df.BT1I != 0]
    df = df[df.BT2I != 0]
    df = df[df.iout != 0]
    
    df=df.dropna()
    print("len 4>",len(df))
    df=df.reset_index(drop=True)
    #with pd.option_context('display.max_columns',None):
    #    print("df head after drop>",df.head())
    neurons = [15,30,50,100]
    activation_function = ['relu','tanh']
    learning_rate = [0.01,0.1,0.2]
    

    param_grid = dict(neurons=neurons,
                  activation_function=activation_function,
                  learning_rate=learning_rate)
    all_mse=[]
    for combination in list(product(*param_grid.values())):
        
        models=[None]*5
        mse=[None]*5
        mae=[None]*5

        i=1
        for train_index, validate_index in kf.split(df):        
            df_train=pd.DataFrame()
            df_val=pd.DataFrame()
            js_train={}
            js_val={}
            df_train_pre=pd.DataFrame(df.loc[train_index])
            df_val_pre=pd.DataFrame(df.loc[validate_index])
            #print("df_train first>",df_train)
            #print("df val first>",df_val)
            aux=[]
            js_train=df_train_pre.to_dict('records',into=OrderedDict)
            #print("js train>",js_train[0])
            js_val=df_val_pre.to_dict('records',into=OrderedDict)
            #js_train=read_json_content(path+str(i)+train_file+str(i))
            #js_val=read_json_content(path+str(i)+val_file+str(i))
        
            
            which='which_topast_fold_'+str(i)
            for att in eval(which):
                js_train=past_commands(js_train,att)
                js_val=past_commands(js_val,att)
                att2=att[0]+'_t'+str(att[1])
                aux.append(att2)
                
            which_tokeep=['GYROX']
            which_tokeep=which_tokeep+aux
            #print(which_tokeep)
        
            #---------Gera os dataframes a partir dos json    
            
            df_train=pd.DataFrame(js_train)
            df_val=pd.DataFrame(js_val)
            
            #---------Mantem no dataframe apenas os atributos alvo e independentes
            
            for col in df_train.columns:
                    if col not in which_tokeep:
                        df_train=df_train.drop(columns=col)
            for col in df_val.columns:
                if col not in which_tokeep:
                    df_val=df_val.drop(columns=col)
        
            
            #with pd.option_context('display.max_columns',None):
            #    print("df train>>",df_train)
            
            
            df_train = df_train.apply(pd.to_numeric, errors='coerce')
            df_train=df_train.dropna()
            df_val = df_val.apply(pd.to_numeric, errors='coerce')
            df_val=df_val.dropna()
            #with pd.option_context('display.max_columns',None):
            #    print("df train>>",df_train)
            print("len df train>",len(df_train))
            print("len df val",len(df_val))
            
            #---------gera numpy para x e y, para train e validation
            
            
            y_train=df_train['GYROX'].values
            y_val=df_val['GYROX'].values
            
            #print(y.shape)
            
            y_train=y_train.reshape(-1,1)
            y_val=y_val.reshape(-1,1)
            
            scaler_y_train = preprocessing.MinMaxScaler()
            scaler_y_val = preprocessing.MinMaxScaler()
            
            y_train = scaler_y_train.fit_transform(y_train)
            y_val = scaler_y_val.fit_transform(y_val)
            
            y_train=y_train.ravel()
            y_val=y_val.ravel()
            #print(y_train)
            #print(y_val)
            
            
            x_train=df_train.drop(columns='GYROX').values
            x_val=df_val.drop(columns='GYROX').values
            
            scaler_x_train = preprocessing.MinMaxScaler()
            scaler_x_val = preprocessing.MinMaxScaler()
            
            x_train = scaler_x_train.fit_transform(x_train)
            x_val = scaler_x_val.fit_transform(x_val)
            
            print(x_train.shape)
            print(x_val.shape)
            if i==1:
                models[i-1] = Sequential()
                models[i-1].add(Dense(41, input_dim=41, kernel_initializer='uniform', activation='tanh'))
                models[i-1].add(Dense(int(combination[0]), kernel_initializer='uniform', activation=str(combination[1])))
                models[i-1].add(Dense(1))
            if i==2:
                models[i-1] = Sequential()
                models[i-1].add(Dense(41, input_dim=41, kernel_initializer='uniform', activation='tanh'))
                models[i-1].add(Dense(int(combination[0]), kernel_initializer='uniform', activation=str(combination[1])))
                models[i-1].add(Dense(1))
            if i==3:
                models[i-1] = Sequential()
                models[i-1].add(Dense(36, input_dim=36, kernel_initializer='uniform', activation='tanh'))
                models[i-1].add(Dense(int(combination[0]), kernel_initializer='uniform', activation=str(combination[1])))
                models[i-1].add(Dense(1))
            if i==4:
                models[i-1] = Sequential()
                models[i-1].add(Dense(50, input_dim=50, kernel_initializer='uniform', activation='tanh'))
                models[i-1].add(Dense(int(combination[0]), kernel_initializer='uniform', activation=str(combination[1])))
                models[i-1].add(Dense(1))
            if i==5:
                models[i-1] = Sequential()
                models[i-1].add(Dense(48, input_dim=48, kernel_initializer='uniform', activation='tanh'))
                models[i-1].add(Dense(int(combination[0]), kernel_initializer='uniform', activation=str(combination[1])))
                models[i-1].add(Dense(1))
            
            
            
            # Compile model
            #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
            opt=SGD(lr=float(combination[2]), momentum=0.9)
            models[i-1].compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error','mae','mape'])
            
            csv_logger = CSVLogger(path_models+"/"+str(combination)+"_Fold"+str(i)+".csv",
                                   append=True, separator=';')
            models[i-1].fit(x_train, y_train,callbacks=[csv_logger], epochs=100, batch_size=200,validation_data=(x_val,y_val), verbose=0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             )
            
            mj = models[i-1].to_json()
            jf = open(path_models+"/"+str(combination)+"_Fold"+str(i)+".json","w")
            jf.write(mj)
            jf.close()
            models[i-1].save_weights(path_models+"/"+str(combination)+"_Fold"+str(i)+".h5")    

            
            metric=models[i-1].evaluate(x_val,y_val)
            mse[i-1]=metric[1]
            mae[i-1]=metric[2]
            
            print("metrica para Fold_"+str(i)+">",mse[i-1])
            i+=1
        
        with open(filename,'a') as f:
            f.write("Metricas para Combinação>\n")
            f.write(str(combination)+'\n')
            f.write("mse>\n")
            f.write(str(mse)+'\n')
            f.write("mae>\n")
            f.write(str(mae)+'\n')
            
            f.write("media mse>\n")
            f.write(str(st.mean(mse))+'\n')
            f.write("std mse>\n")
            f.write(str(st.stdev(mse))+'\n')
            f.write('\n')
            
            f.write("media mae>\n")
            f.write(str(st.mean(mae))+'\n')
            f.write("std mse>\n")
            f.write(str(st.stdev(mae))+'\n')
            f.write('\n')



        print("todas as metricas mse>",mse)
        print("media>",np.nanmean(mse))
        print("std>",st.stdev(mse))
        all_mse.append(np.nanmean(mse))
    with open(filename,'a') as f:
        print("Min mse is:",min(all_mse),file=f)
        print("End of Process",file=f)
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

#41
which_topast_fold_1 = [
    ["mtarg1",1],
    
    ["mtarg2",1],
    ["mtarg2",2],
    
    ["mtarg3",1],
    ["mtarg3",2],
    ["mtarg3",3],
    ["mtarg3",4],
    
    ["roll",1],
            
    ["LACCY",1],
    
    ["LACCZ",1],
    ["LACCZ",2],
    ["LACCZ",3],
    
    ["GYROY",1],
    
    ["GYROZ",1],
    
    ["SC1I",1],    
    ["SC1I",2],    
    ["SC1I",3],    
    ["SC1I",4],    

    
    ["SC3I",1],    
    ["SC3I",2],    
    ["SC3I",3],
    
    ["BT1I",1],    
    ["BT1I",2],
    ["BT1I",3],    
    ["BT1I",4],

    ["vout",1],    
    ["vout",2],    
    ["vout",3],    

    ["iout",1],    
    ["iout",2],    
    ["iout",3],    
    ["iout",4],    

    ["cpuUsage",1],
    
    ["GYROX",1],
    ["GYROX",2],
    ["GYROX",3],
    ["GYROX",4],
    ["GYROX",5],
    ["GYROX",6],
    ["GYROX",7],
    ["GYROX",8]
    
    #["temp",0]
    
    ]
#43
which_topast_fold_2 = [
    ["mtarg1",1],
    
    ["mtarg2",1],
    ["mtarg2",2],
    
    ["mtarg3",1],
    ["mtarg3",2],
    ["mtarg3",3],
    ["mtarg3",4],
    
    ["roll",1],
        
    ["LACCX",1],
    ["LACCX",2],
    ["LACCX",3],
    
    ["LACCY",1],
    
    ["LACCZ",1],
    
    ["GYROY",1],
    ["GYROY",2],
    ["GYROY",3],
    ["GYROY",4],
    
    ["GYROZ",1],
    
    ["SC1I",1],    
    ["SC1I",2],    
    ["SC1I",3],    
    ["SC1I",4],    
    

    
    ["BT1I",1],    
    ["BT1I",2],
    ["BT1I",3],    
    ["BT1I",4],

    ["vout",1],    
    ["vout",2],    
    ["vout",3],    

    ["iout",1],    
    ["iout",2],    
    ["iout",3],    

    ["cpuUsage",1],
    
    ["GYROX",1],
    ["GYROX",2],
    ["GYROX",3],
    ["GYROX",4],
    ["GYROX",5],
    ["GYROX",6],
    ["GYROX",7],
    ["GYROX",8]
    
    #["temp",0]
    
    ]

#36
which_topast_fold_3 = [
    ["mtarg1",1],
    
    ["mtarg2",1],
    ["mtarg2",2],
    
    ["mtarg3",1],
    ["mtarg3",2],
    ["mtarg3",3],
    ["mtarg3",4],
    
    ["roll",1],
        
    ["LACCX",1],
    
    ["LACCY",1],
    
    ["LACCZ",1],
    
    ["GYROY",1],
    
    ["GYROZ",1],
    
    ["SC1I",1],    
    ["SC1I",2],    
    ["SC1I",3],    
    ["SC1I",4],    
        
    ["BT1I",1],    
    ["BT1I",2],
    ["BT1I",3],    
    ["BT1I",4],

    ["vout",1],    
    ["vout",2],    
    ["vout",3],    

    ["iout",1],    
    ["iout",2],    
    ["iout",3],    

    ["cpuUsage",1],
    
    ["GYROX",1],
    ["GYROX",2],
    ["GYROX",3],
    ["GYROX",4],
    ["GYROX",5],
    ["GYROX",6],
    ["GYROX",7],
    ["GYROX",8]
    
    #["temp",0]
    
    ]
#52
which_topast_fold_4 = [
    ["mtarg1",1],
    
    ["mtarg2",1],
    ["mtarg2",2],
    ["mtarg2",3],
    ["mtarg2",4],
    ["mtarg2",5],
    ["mtarg2",6],
    ["mtarg2",7],
    ["mtarg2",8],
    ["mtarg2",9],
    ["mtarg2",10],
    ["mtarg2",11],
    ["mtarg2",12],
    ["mtarg2",13],
    ["mtarg2",14],
    ["mtarg2",15],

    
    ["mtarg3",1],
    ["mtarg3",2],
    ["mtarg3",3],
    ["mtarg3",4],
    
    ["roll",1],
        
    ["LACCX",1],
    
    ["LACCY",1],
    
    ["LACCZ",1],
    
    
    ["GYROZ",1],
    
    ["SC1I",1],    
    ["SC1I",2],    
    ["SC1I",3],    
    ["SC1I",4],    

    
    ["SC3I",1],    
    
    ["BT1I",1],    
    ["BT1I",2],
    ["BT1I",3],    
    ["BT1I",4],


    ["vout",1],    
    ["vout",2],    
    ["vout",3],    

    ["iout",1],    
    ["iout",2],    
    ["iout",3],    
    ["iout",4],    

    ["cpuUsage",1],
    
    ["GYROX",1],
    ["GYROX",2],
    ["GYROX",3],
    ["GYROX",4],
    ["GYROX",5],
    ["GYROX",6],
    ["GYROX",7],
    ["GYROX",8]
    
    #["temp",0]
    
    ]

#48
which_topast_fold_5 = [
    ["mtarg1",1],
    
    ["mtarg2",1],
    ["mtarg2",2],
    ["mtarg2",3],
    ["mtarg2",4],
    ["mtarg2",5],
    ["mtarg2",6],
    ["mtarg2",7],
    ["mtarg2",8],
    ["mtarg2",9],
    ["mtarg2",10],
    ["mtarg2",11],
    ["mtarg2",12],
    ["mtarg2",13],
    ["mtarg2",14],
    ["mtarg2",15],
    
    ["mtarg3",1],
    ["mtarg3",2],
    ["mtarg3",3],
    ["mtarg3",4],
    
    ["roll",1],
        
    ["LACCX",1],
    
    ["LACCY",1],
    
    ["LACCZ",1],
    
    ["GYROY",1],
    
    ["GYROZ",1],
    
    ["SC1I",1],    
    ["SC1I",2],    
    ["SC1I",3],    
    ["SC1I",4],    

    ["SC2I",1],    
    ["SC2I",2],
    
    ["SC3I",1],    
    ["SC3I",2],    
    ["SC3I",3],
    
    ["BT2I",1],    
    ["BT2I",2],    
    ["BT2I",3],    
    ["BT2I",4],    


    ["cpuUsage",1],
    
    ["GYROX",1],
    ["GYROX",2],
    ["GYROX",3],
    ["GYROX",4],
    ["GYROX",5],
    ["GYROX",6],
    ["GYROX",7],
    ["GYROX",8]
    
    #["temp",0]
    
    ]



which_tokeep_0 = [
    "mtarg1",
    "mtarg2", #tirado 3
    "mtarg3",
    #["deapth",7], #comecou sem
    "roll",
    "LACCX",
    "LACCY",
    "LACCZ", #tirado 9 ->0.700
    "GYROX", #tirado 4
    "GYROY", #tirado 6 ->0.703
    "SC1I", #tirado 8 ->0.701
    "SC2I", #tirado 7 ->0.702
    "SC3I",
    "BT1I", #tirado 5
    "BT2I", #tirado 10 ->0.699
    "vout",  #tirado 11 ->0.698
    "iout",  #tirado 1
    "timestamp",
    "has_elapsed",
    "cpuUsage", 
    "GYROZ"
    #"temp",  #comecou sem
    
    ]

if (__name__=='__main__'):
    main();

