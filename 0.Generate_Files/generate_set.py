#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:33:22 2019

@author: felipeadachi
"""
import pickle
import numpy as np
from sklearn.model_selection import KFold
import json
import pandas as pd
from collections import OrderedDict
import collections
import os

db_json='/home/felipeadachi/Dados/DB-Train-JSON'
path_out='/home/felipeadachi/Dados/Kfolds_cpu'

def main():
    df=pd.DataFrame()
    with open(db_json+'/DB-Train','rb') as f:
        json_content=pickle.load(f)
    
    df=pd.DataFrame(json_content)
    print("len1>",len(df))
    kf = KFold(n_splits=5,shuffle=False,random_state=2)
    
    
    for col in df.columns:
        if col not in which_tokeep:
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
    df = df[df.cpuUsage != 1] ################Esta linha foi adicionada
    #para gerar os arquivos em kfolds-cpu
    print("len 3>",len(df))
    df=df.dropna()
    print("len 4>",len(df))
    df=df.reset_index(drop=True)
    #with pd.option_context('display.max_columns',None):
    #    print("df head after drop>",df.head())
    
    i=1
    for train_index, validate_index in kf.split(df):        
        #print("train_index",train_index)
        #print("validate_index",validate_index)
        trainDF = pd.DataFrame(df.loc[train_index])
        validateDF = pd.DataFrame(df.loc[validate_index])
        validateDF=validateDF.dropna()
        trainDF = trainDF.dropna()
        #print("validateDF>>>>>>>>",validateDF)
        js_val=validateDF.to_dict('records',into=OrderedDict)
        js_val=has_elapsed(js_val)
        js_tr=trainDF.to_dict('records',into=OrderedDict)
        js_tr=has_elapsed(js_tr)
        print("iteração",i)        
        #print(js_val)
        #valDF2=pd.DataFrame(js_val)
        #print("valDF2>>>>>>>",valDF2)
        write_json_content(path_out + '/Fold_'+str(i) + '/Train_' + str(i),js_tr)
        write_json_content(path_out + '/Fold_'+str(i) + '/Validate_' + str(i),js_val)
        
        i+=1
                

#---------------------#---------------------#---------------------#---------------------#---------------------#

def read_json_content(filename):
    with open(filename, 'r') as f:
        return json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(f.read(os.stat(filename).st_size))

#---------------------#---------------------#---------------------#---------------------#---------------------#

def write_json_content(filename, content):
    with open(filename, 'w') as f:
        f.write(json.dumps(content,indent=4))

#---------------------#---------------------#---------------------#---------------------#---------------------#


def has_elapsed(content):
    
    
    i=0
    
    for element in content:    
        if i==0:
            element['has_elapsed']=1
        if i >= 1:
            if (element['timestamp']-content[i-1]['timestamp'])>=500:
                element['has_elapsed']=1
            else:
                element['has_elapsed']=0
        i+=1
    return content

#---------------------#---------------------#---------------------#---------------------#---------------------#



which_tokeep = [
    "mtarg1",
    "mtarg2", 
    "mtarg3",
    "roll",
    "pitch", 
    "LACCX",
    "LACCY",
    "LACCZ", 
    "GYROX", 
    "GYROY", 
    "SC1I", 
    "SC2I", 
    "SC3I",
    "BT1I",
    "BT2I",
    "vout",
    "iout",  
    "timestamp",
    "has_elapsed",
    "cpuUsage", ##############Esta linha foi adicionada para os arquivos
    #em kfolds_cpu
    "GYROZ"
    
    ]

if(__name__ == '__main__'):
    main();