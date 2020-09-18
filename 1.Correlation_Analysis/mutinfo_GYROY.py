#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:50:43 2019

@author: felipeadachi
"""

import pickle
import numpy as np
from sklearn.model_selection import KFold
import json
import pandas as pd
import collections
from collections import OrderedDict
import os
from sklearn import preprocessing
import copy
from matplotlib import pyplot as plt
import pyper as pr


db_json='/home/felipeadachi/Dados'
path='/home/felipeadachi/Dados/Kfolds_cpu'


#file_name='2.s1.v3.controle.dpfreq.13:40.json'
#path='/home/felipeadachi/Dados/corrteste'
        
past_to_get=list(range(1,26))

#db_json='/home/felipeadachi/Dados/DB-Train-JSON'
    

def main():
    df=pd.DataFrame()
    with open(db_json+'/DB-Train','rb') as f:
        json_content=pickle.load(f)

    aux=[]
    #json_content=read_json_content(db_json)
    df=pd.DataFrame(json_content)
    #df=pd.DataFrame(json_content['0']['data'])
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
    #print("len 2>",len(df))
    df = df[df.cpuUsage != 1]
    
    
    df=df.dropna()
    df=df.reset_index(drop=True)
    #df=df.reindex()
    #with pd.option_context('display.max_rows',None):
    #    print(df)
    i=1
    for train_index, validate_index in kf.split(df):
        index_corr=np.reshape(which_topast,(1,-1))
        cross_corr=pd.DataFrame(columns=['max_corr','timelag'],index=index_corr[0])

        print("iteração",i)        
        #print("train_index",train_index)
        #print("validate_index",validate_index)
        trainDF = pd.DataFrame(df.loc[train_index])
        validateDF = pd.DataFrame(df.loc[validate_index])
        validateDF=validateDF.dropna()
        #print("validateDF>>>>>>>>",validateDF)
        js_val=validateDF.to_dict('records',into=OrderedDict)
        js_val=has_elapsed(js_val)
        js_tr=trainDF.to_dict('records',into=OrderedDict)
        js_tr=has_elapsed(js_tr)
            
        
        #print(js_val)
        #valDF2=pd.DataFrame(js_val)
        #trDF2=pd.DataFrame(js_tr)
        for att in which_topast:
            df_2=pd.DataFrame()
            which_tokeep=[]
            aux=[]
            js_tr2=past_commands(js_tr,att)
            #print("new attribute>>>>>>>")
            #print(js_tr2[0])
            for past in past_to_get:
                att2=att[0]+'_t'+str(past)
                aux.append(att2)
            which_tokeep=['GYROY','timestamp']
            which_tokeep=which_tokeep+att
            which_tokeep=which_tokeep+aux
            print("whichtokeep",which_tokeep)
            
            df_2=df_2.append(pd.DataFrame(js_tr2))
            
            for col in df_2.columns:
                if col not in which_tokeep:
                    df_2=df_2.drop(columns=col)
            df_2=df_2.dropna()
            
            df_2=df_2.drop(['timestamp'],axis=1)
            x=df_2.values #returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df3 = pd.DataFrame(x_scaled)
            
            df3.columns=df_2.columns
            #df3=df3.set_index('timestamp')
        
            #print("df3>",df3)
            r=pr.R(use_pandas=True) 
            r.assign("rdf2", df3)
            r('library(infotheo)')
            r('dat<-discretize(rdf2)')
            r('I<-mutinformation(dat)')
            mutual_Info=r.I
            corr=pd.DataFrame(mutual_Info)
            corr.columns=df3.columns
            corr=corr.set_index(df3.columns)
            #corr=df3.corr(method='spearman')
            maxcorr,indx=findmax(corr)
            #print("max correlation is>",corr['GYROZ'][indx],"for timelag>",indx)
            timelag=indx.split("_")
            cross_corr.loc[att[0]]=[corr['GYROY'][indx],timelag[-1]]
        
            #print(corr['GYROZ'])
            filename=''+att[0]+''
            corr['GYROY'].drop(index='GYROY').to_csv(path+'/Fold_'+str(i)+'/mutinfo_analysis/GYROY/'+filename+'.csv')
            arr=corr['GYROY'].drop(index='GYROY').as_matrix()
            plt.plot(arr)
            plt.savefig(path+'/Fold_'+str(i)+'/mutinfo_analysis/GYROY/'+filename+'.png')
            #plt.show() 
            #print("corr>",df3.corr())
        cross_corr.to_csv(path+'/Fold_'+str(i)+'/mutinfo_analysis/GYROY/mutinfo_GYROY.csv')





        i+=1
        #print("valDF2>>>>>>>",valDF2)
        #print(validateDF)
        #y=trainDF['GYROZ'].values
        #print(y)
        #print(y.shape)
        #y=y.reshape(-1,1)
        
        #min_max_scaler_y = preprocessing.MinMaxScaler()
        #y = min_max_scaler_y.fit_transform(y)
        #y=y.ravel()
        #print(list(df))
        print("End of process execution")


    
    

#---------------------#---------------------#---------------------#---------------------#---------------------#

def read_json_content(filename):
    with open(filename, 'r') as f:
        return json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(f.read(os.stat(filename).st_size))

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

#---------------------#---------------------#---------------------#---------------------#---------------------#

def past_commands(content,att):
    
    data_list=copy.deepcopy(content)
    
    i=0
    
        
    while (i<len(data_list)):
        
        #get_token retorna o número de vezes permitido para pegar os comandos passados, com base nos flags "has_elapsed" de cada timestep
        token=get_token(i,data_list)
        #print("token for timestep:",i,"is:",token)
        data=data_list[i]
        tkn={}
        
        tkn[att[0]]=token
                
        for past in past_to_get:
            
            if i<past:
                
                data[att[0]+'_t'+str(past)]=np.nan

            else:
                    
                    
                try:
                    if tkn[att[0]]<=0:
                        data[att[0]+'_t'+str(past)]=np.nan
                    else:
                        data[att[0]+'_t'+str(past)]=data_list[i-past][att[0]]
                        tkn[att[0]]-=1
                        #print("token for:",motor,"in tstep:",i,"is:",tkn[motor[0]])
                except KeyError:
                    data[att[0]+'_t'+str(past)]=np.nan
                    tkn[att[0]]-=1
        i+=1
    #print("data_list>",data_list[0])
    #print("content>",content[0])
    return data_list

#---------------------#---------------------#---------------------#---------------------#---------------------#

def moving_average(content,att):
    sum_=0
    data_list=content
    
    i=0
    isnan=False
    while (i<len(data_list)):
        data=data_list[i]
        for past in past_to_get:
            for tn in range(past):
                name=att[0]+'_t'+str(tn+1)
                if data[name] == np.nan or data[name] == 'NaN':
                    isnan=True
                else:
                    sum_=sum_+float(data[name])
            if not isnan:
                data[att[0]+'_m'+str(past)]=sum_/past
            else:
                data[att[0]+'_m'+str(past)]=np.nan
            isnan=False
            sum_=0
        i+=1
    return content

#---------------------#---------------------#---------------------#---------------------#---------------------#


def get_token(i,data_list):
    j=0
    token=0
    while j<max(past_to_get):
        if data_list[i-j]['has_elapsed'] == 0:
            token+=1
        else:
            return token
        
        j+=1
    return token


#---------------------#---------------------#---------------------#---------------------#---------------------#

def findmax(df):
    maxcorr=0
    indx=''
    for index, row in df.iterrows():
        if abs(row['GYROY']) > maxcorr and abs(row['GYROY']) < 1: 
            maxcorr=abs(row['GYROY'])
            indx=index
    return maxcorr,indx        

#---------------------#---------------------#---------------------#---------------------#---------------------#


which_tokeep_0 = [
    "mtarg1",
    "mtarg2", #tirado 3
    "mtarg3",
    "deapth", #comecou sem
    "roll",
    "pitch",  #tirado 2
    "deapth",
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
    "GYROZ",
    "temp",  #comecou sem
    "cpuUsage"
    
    ]

which_topast=[
    ['mtarg1'],
    ['mtarg2'],
    ['mtarg3'],
    ['roll'],
    ['pitch'],
    #['yaw'],
    ['deapth'],
    ['LACCX'],
    ['LACCY'],
    ['LACCZ'],
    ['GYROX'],
    ['GYROZ'],
    ['SC1I'],
    ['SC2I'],
    ['SC3I'],
    ['BT1I'],
    ['BT2I'],
    ['vout'],
    ['iout'],
    ['temp'],
    #['deapth_delta']
    #['alps'],
    ['cpuUsage']
    #['AVCC'],
    #['GYROZ']
    
    ]


if(__name__ == '__main__'):
    main();
