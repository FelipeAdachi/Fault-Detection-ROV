#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:10:05 2019

@author: felipeadachi
"""

path='/home/felipeadachi/Dados/Kfolds_cpu'
train_file='/Whole_Train.json'
test_file='/Whole_Test.json'
path_models = '/home/felipeadachi/svn-rov/smartx/software/Machine_learning_for_ROV_fault_detection/4.Train_Final_Models'
import json
import os
import collections
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pickle

def get_which_topast(target):
    if target == "GYROZ":
        which_topast.remove(['GYROZ',1])
    if target == "GYROX":
        which_topast.remove(['GYROX',1])
        which_topast.remove(['pitch',1])

    if target == "GYROY":
        which_topast.remove(['GYROY',1])
        which_topast.remove(['roll',1])

    return which_topast


def main():
    target = 'GYROY'
    df_train=pd.DataFrame()
    df_val=pd.DataFrame()
    
    aux=[]
    
    js_train=read_json_content(path+train_file)
    js_val=read_json_content(path+test_file)

    #---------Gera os atributos passados de acordo com which_topast    
    which = get_which_topast(target)
    for att in which:
        js_train=past_commands(js_train,att)
        js_val=past_commands(js_val,att)
        att2=att[0]+'_t'+str(att[1])
        aux.append(att2)
    
    which_tokeep=[target]
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
    
    
    y_train=df_train[target].values
    y_val=df_val[target].values
    
    #print(y.shape)
    
    y_train=y_train.reshape(-1,1)
    y_val=y_val.reshape(-1,1)
    
    scaler_y_train = preprocessing.MinMaxScaler()
    scaler_y_val = preprocessing.MinMaxScaler()
    
    
    scaler_y_train.fit(y_train)
    
    
    
    file = open('BL_y.pickle', 'wb')
    pickle.dump(scaler_y_train,file)
    file.close()
    
    y_train = scaler_y_train.transform(y_train)
    
    y_val = scaler_y_val.fit_transform(y_val)
    
    y_train=y_train.ravel()
    y_val=y_val.ravel()
    print(y_train)
    print(y_val)
    
    
    x_train=df_train.drop(columns=target).values
    x_val=df_val.drop(columns=target).values
    
    scaler_x_train = preprocessing.MinMaxScaler()
    scaler_x_val = preprocessing.MinMaxScaler()
    
    scaler_x_train.fit(x_train)
    
    #scalerx=min_max_scaler_x.fit(x_train)

    
    file = open('BL_x.pickle', 'wb')
    pickle.dump(scaler_x_train,file)
    file.close()

    x_train = scaler_x_train.transform(x_train)
    x_val = scaler_x_val.fit_transform(x_val)
    
    print(x_train.shape)
    print(x_val.shape)
    #------Modelling
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    import math
    poly_model = None
    poly_model = make_pipeline(PolynomialFeatures(degree=2),
                       Ridge(alpha=10))
    
    poly_model.fit(x_train, y_train)
    y_predict = poly_model.predict(x_val)

    from joblib import dump
    filenme = path_models+"/BL_"+str(target)+".pickle"
    dump(poly_model,filenme)
    
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_val, y_predict)
    rmse = math.sqrt(mse)
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_val, y_predict)
    filename = path_models+"/log_final_BL.txt"
    with open(filename,'a') as f:
        f.write("Metricas>\n")
        f.write("mse>\n")
        f.write(str(mse)+'\n')
        f.write("rmse>\n")
        f.write(str(rmse)+'\n')
        f.write("mae>\n")
        f.write(str(mae)+'\n')

    #------Modelling
    
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
    ["mtarg2",1],  
    ["mtarg3",1],
    
    ["roll",1],
    ["pitch",1],

    ["LACCX",1],          
    ["LACCY",1],  
    ["LACCZ",1],
    
    ["GYROX",1],
    ["GYROY",1],  
    ["GYROZ",1],
    
    ["SC1I",1],    
    ["SC2I",1],    
    ["SC3I",1],
    
    
    ["BT1I",1],    
    ["BT2I",1],    

    ["vout",1],    
    ["iout",1],    

    ["cpuUsage",1],
        
    ]


if (__name__=='__main__'):
    main();
