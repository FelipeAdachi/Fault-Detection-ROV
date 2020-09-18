

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Imports

"""
Created on Thu Jun 13 12:00:05 2019

@author: felipeadachi
"""

from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
set_random_seed(2)


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
import math


#%% Global Variables
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




which_tokeep_0 = [
    "mtarg1",
    "mtarg2", #tirado 3
    "mtarg3",
    "roll",
    "pitch",
    "LACCX",
    "LACCY",
    "LACCZ",
    "GYROX",
    "GYROY",
    "GYROZ",
    "SC1I",
    "SC2I",
    "SC3I",
    "BT1I",
    "BT2I",
    "vout",
    "iout",
    "timestamp",
    "has_elapsed",
    "cpuUsage", 
    "GYROZ"
    
    ]



#%% Defs

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




#%% Preproc
targets = ['GYROX','GYROY','GYROZ']
path='/home/felipeadachi/Dados/Kfolds_cpu/Fold_'
train_file='/Train_'
val_file='/Validate_'
db_json='/home/felipeadachi/Dados/DB-Train-JSON'

for target in targets:
    print("TARGET>>>",target)
    filename = '/home/felipeadachi/Dados/Kfolds_cpu/Models_GridSearch/' + target + '/Baseline/log_baseline.txt'
    which = get_which_topast(target)
    path_models='/home/felipeadachi/Dados/Kfolds_cpu/Models_GridSearch/'+str(target)+'/Baseline'
    
        
    
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
    
    #%% Grid Search
    alpha = [0,0.01,0.1,1,10,100,1000,10000]
    degree = [1,2,3]
    
    
    param_grid = dict(alpha=alpha,
                  degree=degree)
    all_mse=[]
    for combination in list(product(*param_grid.values())):
        poly_model = [None]*5
        models=[None]*5
        mse=[None]*5
        mae=[None]*5
        rmse=[None]*5
    
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
        
            
            for att in which:
                js_train=past_commands(js_train,att)
                js_val=past_commands(js_val,att)
                att2=att[0]+'_t'+str(att[1])
                aux.append(att2)
                
            which_tokeep=[target]
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
            
            
            y_train=df_train[target].values
            y_val=df_val[target].values
            
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
            
            
            x_train=df_train.drop(columns=target).values
            x_val=df_val.drop(columns=target).values
            
            scaler_x_train = preprocessing.MinMaxScaler()
            scaler_x_val = preprocessing.MinMaxScaler()
            
            x_train = scaler_x_train.fit_transform(x_train)
            x_val = scaler_x_val.fit_transform(x_val)
            
            print(x_train.shape)
            print(x_val.shape)
        
    
    #####------Modelling
            
            #-Linear regression------------------------------
            #from sklearn.linear_model import LinearRegression
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import PolynomialFeatures
            poly_model[i-1] = make_pipeline(PolynomialFeatures(degree=combination[1]),
                               Ridge(alpha=combination[0]))
            
            poly_model[i-1].fit(x_train, y_train)
            y_predict = poly_model[i-1].predict(x_val)
    #        for pred,real in zip(y_predict,y_val):
    #            print("Predict>",pred)
    #            print("Real>",real)
    #            print("#---------------#---------#-----------")
    #        
            from joblib import dump
            filenme = path_models+"/"+str(combination)+"_Fold"+str(i)+".pickle"
            dump(poly_model[i-1],filenme)
            
            from sklearn.metrics import mean_squared_error
            mse[i-1] = mean_squared_error(y_val, y_predict)
            rmse[i-1] = math.sqrt(mse[i-1])
            from sklearn.metrics import mean_absolute_error
            mae[i-1] = mean_absolute_error(y_val, y_predict)
            
            
            print("Finished loading model>>>>",i)
            #-Linear regression------------------------------
            i+=1
    
        with open(filename,'a') as f:
            f.write("Metricas para Combinação>\n")
            f.write(str(combination)+'\n')
            f.write("mse>\n")
            f.write(str(mse)+'\n')
            f.write("rmse>\n")
            f.write(str(rmse)+'\n')
    
            f.write("mae>\n")
            f.write(str(mae)+'\n')
            
            f.write("media mse>\n")
            f.write(str(st.mean(mse))+'\n')
    
            f.write("std mse>\n")
            f.write(str(st.stdev(mse))+'\n')
            f.write('\n')
    
            f.write("media rmse>\n")
            f.write(str(st.mean(rmse))+'\n')
    
            f.write("std rmse>\n")
            f.write(str(st.stdev(rmse))+'\n')
            f.write('\n')
            
            f.write("media mae>\n")
            f.write(str(st.mean(mae))+'\n')
            f.write("std mae>\n")
            f.write(str(st.stdev(mae))+'\n')
            f.write('\n')
    
            print("todas as metricas mse>",mse)
            print("media>",np.nanmean(mse))
            print("std>",st.stdev(mse))
            all_mse.append(np.nanmean(mse))
    with open(filename,'a') as f:
        print("Min mse is:",min(all_mse),file=f)
        print("End of Process",file=f)
print("End of Process")    
        
#        models[i-1] = Sequential()
#        models[i-1].add(Dense(40, input_dim=40, kernel_initializer='uniform', activation='tanh'))
#        models[i-1].add(Dense(int(combination[0]), kernel_initializer='uniform', activation=str(combination[1])))
#        models[i-1].add(Dense(1))
#        
#        
#        # Compile model
#        #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
#        opt=SGD(lr=float(combination[2]), momentum=0.9)
#        models[i-1].compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error','mae','mape'])
#        
#        csv_logger = CSVLogger(path_models+"/"+str(combination)+"_Fold"+str(i)+".csv",
#                               append=True, separator=';')
#        models[i-1].fit(x_train, y_train,callbacks=[csv_logger], epochs=100, batch_size=200,validation_data=(x_val,y_val), verbose=0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             )
#        
#        mj = models[i-1].to_json()
#        jf = open(path_models+"/"+str(combination)+"_Fold"+str(i)+".json","w")
#        jf.write(mj)
#        jf.close()
#        models[i-1].save_weights(path_models+"/"+str(combination)+"_Fold"+str(i)+".h5")    

        
#        metric=models[i-1].evaluate(x_val,y_val)
#        mse[i-1]=metric[1]
#        mae[i-1]=metric[2]
#        
#        print("metrica para Fold_"+str(i)+">",mse[i-1])
#        i+=1
#    
#    with open(filename,'a') as f:
#        f.write("Metricas para Combinação>\n")
#        f.write(str(combination)+'\n')
#        f.write("mse>\n")
#        f.write(str(mse)+'\n')
#        f.write("mae>\n")
#        f.write(str(mae)+'\n')
#        
#        f.write("media mse>\n")
#        f.write(str(st.mean(mse))+'\n')
#        f.write("std mse>\n")
#        f.write(str(st.stdev(mse))+'\n')
#        f.write('\n')
#        
#        f.write("media mae>\n")
#        f.write(str(st.mean(mae))+'\n')
#        f.write("std mse>\n")
#        f.write(str(st.stdev(mae))+'\n')
#        f.write('\n')
#
#
#
#    print("todas as metricas mse>",mse)
#    print("media>",np.nanmean(mse))
#    print("std>",st.stdev(mse))
#    all_mse.append(np.nanmean(mse))
#with open(filename,'a') as f:
#    print("Min mse is:",min(all_mse),file=f)
#    print("End of Process",file=f)
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
##----------------NN model       
#---------------------#---------------------#---------------------#---------------------#---------------------#


#%%
    
#import pickle
#
#filename = 'ytrain.pickle'
#outfile = open(filename,'wb')
#pickle.dump(y_train,outfile)
#outfile.close()
