#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:21:40 2019

@author: felipeadachi
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
import json
import collections
import pyper as pr

kfd='Train_5'

def main():

    for k in range(1,6):
        file='/home/felipeadachi/Dados/Kfolds_cpu/Fold_'+str(k)+'/Train_'+str(k)
        df=pd.DataFrame()
        aux=[]
        js=read_json_content(file)
        #print("jssss>",js[1])
        which='which_topast_'+str(k)
        for att in eval(which):
            js2=past_commands(js,att)
            att2=att[0]+'_t'+str(att[1])
            aux.append(att2)
        which_tokeep=['GYROX']
        which_tokeep=which_tokeep+aux
        #print(which_tokeep)
        
        df=df.append(pd.DataFrame(js2))
        
        for col in df.columns:
                if col not in which_tokeep:
                    df=df.drop(columns=col)
        
        df=df.dropna()
        
        df = df.apply(pd.to_numeric, errors='coerce')
        
        #df['current_t1']=df['SC1I_t1']+df['SC3I_t1']+df['BT1I_t1']+df['BT2I_t1']
        
        
        
        #df=df.drop(columns='SC1I_t1')
        #df=df.drop(columns='SC3I_t1')
        #df=df.drop(columns='BT1I_t1')
        #df=df.drop(columns='BT2I_t1')
        
        y=df['GYROX'].values
        #print(y.shape)
        y=y.reshape(-1,1)
        
        min_max_scaler_y = preprocessing.MinMaxScaler()
        y = min_max_scaler_y.fit_transform(y)
        y=y.ravel()
        #print("x",len(x))
        df=df.drop(columns='GYROX')
        #df=df.drop(columns='alps_t7')
        #df=df.drop(columns='cpuUsage_t5')
        #df=df.drop(columns='AVCC_t8')
        print(list(df))
        
        #print("Sum esc>",df['SC13I_t1'])
        #print("Sum btt>",df['BTI_t1'])
        
        x=df.values
        min_max_scaler_x = preprocessing.MinMaxScaler()
        x = min_max_scaler_x.fit_transform(x)
        #print("xshape>",x.shape)    
        #print("leny",y.shape)
        #print(y.shape)
        #print(np.ones([96883,1]).shape)
        
        dfx = pd.DataFrame(x)
        dfy = pd.DataFrame(y)
        dfy.columns=['GYROX']
        #df3=df3.assign(timestamp=df['timestamp'])
            
        dfx.columns=df.columns
        dft = pd.concat([dfx,dfy], axis=1)
        #print("dft>",dft)
        r=pr.R(use_pandas=True) 
        r.assign("rdf2", dft)
        r('library(CORElearn)')            
        r('cat(versionCore(),"\n")')
        r('estReliefF <- attrEval("GYROX", rdf2,estimator="ReliefFexpRank", ReliefIterations=0)')
        
        estReliefF=r.estReliefF
        with open('Relief_Fold_'+str(k)+'_GYROX_nopitch','a') as f:
            f.write('estimator="ReliefFexpRank", ReliefIterations=0. Neighbours=70, sigma=20\n')
            f.write(str(list(df))+"\n")
            f.write(str(estReliefF)+"\n")
        print("Iteração"+str(k)+"\n")
    print("End of process")
        #print("estrelieff>",estReliefF)
        #r('profiles <- ordDataGen(200)')
        #r('est <- ordEval(class ~ ., profiles, ordEvalNoRandomNormalizers=100)')
        #est=r.est
        #print("est>",est)
                
    
    
            
    
    
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

def read_json_content(filename):
    with open(filename, 'r') as f:
        return json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(f.read(os.stat(filename).st_size))

#---------------------#---------------------#---------------------#---------------------#---------------------#




which_topast_1 = [
    ["mtarg1",1],
    ["mtarg2",2],
    ["mtarg3",4],
    #["deapth",0],
    ["roll",1],
    ["LACCX",3],
    ["LACCY",1],
    ["LACCZ",3],
    ["GYROZ",1],
    ["GYROY",1],
    ["SC1I",4],
    ["SC2I",2],
    ["SC3I",3],
    ["BT1I",4],
    ["BT2I",4],
    ["vout",3],
    ["iout",4],
    ["cpuUsage",1]
    #["temp",0]
    
    ]

which_topast_2 = [
    ["mtarg1",1],
    ["mtarg2",2],
    ["mtarg3",4],
    #["deapth",0],
    ["roll",1],
    ["LACCX",3],
    ["LACCY",1],
    ["LACCZ",1],
    ["GYROZ",1],
    ["GYROY",1],
    ["SC1I",4],
    ["SC2I",2],
    ["SC3I",4],
    ["BT1I",4],
    ["BT2I",4],
    ["vout",3],
    ["iout",3],
    ["cpuUsage",1]
    #["temp",0]
    
    ]

which_topast_3 = [
    ["mtarg1",1],
    ["mtarg2",2],
    ["mtarg3",4],
    #["deapth",0],
    ["roll",1],
    ["LACCX",1],
    ["LACCY",1],
    ["LACCZ",1],
    ["GYROZ",1],
    ["GYROY",1],
    ["SC1I",4],
    ["SC2I",2],
    ["SC3I",4],
    ["BT1I",4],
    ["BT2I",4],
    ["vout",3],
    ["iout",3],
    ["cpuUsage",1]
    #["temp",0]
    
    ]

which_topast_4 = [
    ["mtarg1",1],
    ["mtarg2",25],
    ["mtarg3",4],
    #["deapth",0],
    ["roll",1],
    ["LACCX",1],
    ["LACCY",1],
    ["LACCZ",1],
    ["GYROZ",2],
    ["GYROY",1],
    ["SC1I",4],
    ["SC2I",2],
    ["SC3I",1],
    ["BT1I",4],
    ["BT2I",4],
    ["vout",3],
    ["iout",4],
    ["cpuUsage",1]
    #["temp",0]
    
    ]

which_topast_5 = [
    ["mtarg1",1],
    ["mtarg2",25],
    ["mtarg3",4],
    #["deapth",0],
    ["roll",1],
    ["LACCX",1],
    ["LACCY",1],
    ["LACCZ",1],
    ["GYROZ",1],
    ["GYROY",1],
    ["SC1I",4],
    ["SC2I",2],
    ["SC3I",3],
    ["BT1I",4],
    ["BT2I",4],
    ["vout",3],
    ["iout",4],
    ["cpuUsage",1]
    #["temp",0]
    
    ]



if (__name__=='__main__'):
    main();
