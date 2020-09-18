#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:45:23 2019

@author: felipeadachi
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import arff
import os
import json
import collections
import pickle
import argparse

path='/home/felipeadachi/Dados/serverlab/Whole_train'


parser = argparse.ArgumentParser(description="Stepwise selection, using timeshifted values given by correlation analysis.")
required = parser.add_argument_group('required named arguments')

required.add_argument('-t','--target',choices=['GYROX','GYROY','GYROZ'], help='Target feature. Either GYROX,GYROY or GYROZ',required=True)

arguments = parser.parse_args()
args = vars(parser.parse_args())

target=arguments.target
print(target)



def main():
    correlation_results='/home/felipeadachi/Dados/serverlab/Fold_1/mutinfo_analysis/'+str(target)+'/mutinfo_'+str(target)+'.csv'
    
    
    db_json='/home/felipeadachi/Dados/DB-Train-JSON'

    df=pd.DataFrame()
    aux=[]
    
    with open(db_json+'/DB-Train','rb') as f:
        js=pickle.load(f)
    
    which_topast=get_corr_results(correlation_results)
    
    for att in which_topast:
        js2=past_commands(js,att)
        att2=att[0]+'_t'+str(att[1])
        aux.append(att2)
    which_tokeep=[target]
    which_tokeep=which_tokeep+aux
    #print(which_tokeep)
    
    df=df.append(pd.DataFrame(js2))
    
    for col in df.columns:
            if col not in which_tokeep:
                df=df.drop(columns=col)
    
    df=df.dropna()
    
    y=df[target].values
    #print(y.shape)
    y=y.reshape(-1,1)
    
    min_max_scaler_y = preprocessing.MinMaxScaler()
    y = min_max_scaler_y.fit_transform(y)
    y=y.ravel()
    #print("x",len(x))
    df=df.drop(columns=target)
    #print(list(df))
    x=df.values
    min_max_scaler_x = preprocessing.MinMaxScaler()
    x = min_max_scaler_x.fit_transform(x)

    #print("xshape>",x.shape)    
    #print("leny",y.shape)
    #print(y.shape)
    #print(np.ones([96883,1]).shape)
    
    dfx = pd.DataFrame(x)
    dfy = pd.DataFrame(y)
    dfy.columns=[target]
    #df3=df3.assign(timestamp=df['timestamp'])
        
    dfx.columns=df.columns
    dft = pd.concat([dfx,dfy], axis=1)
    print("dft>",dft)
    
    path_out=path+'/weka/'
    data_file=str(target)
    arff.dump(path_out+data_file+'.arff'
      , dft.values
      , relation=data_file
      , names=dft.columns)
    
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
def get_corr_results(path):
    
    data=pd.read_csv(path)
    which_topast = []

    for i in data.index:
        if str(data.loc[i]['timelag'])==str(data.loc[i][0]):
            print(data.loc[i][0])
            toappend=[data.loc[i][0],1]
            which_topast.append(toappend)
        else:    
            toappend=[data.loc[i][0],int(data.loc[i]['timelag'][1:])]
            which_topast.append(toappend)
            print(data.loc[i][0])
        
    return which_topast



#---------------------#---------------------#---------------------#---------------------#---------------------#

def read_json_content(filename):
    with open(filename, 'r') as f:
        return json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(f.read(os.stat(filename).st_size))

#---------------------#---------------------#---------------------#---------------------#---------------------#


if (__name__=='__main__'):
    main()
