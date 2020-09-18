#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:21:07 2019

@author: felipeadachi
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
import json
import collections
import pyper as pr
import pickle
import argparse
import re



parser = argparse.ArgumentParser(description="Relieff selection, using timeshifted values given by correlation analysis.")
required = parser.add_argument_group('required named arguments')

required.add_argument('-t','--target',choices=['GYROX','GYROY','GYROZ'], help='Target feature. Either GYROX,GYROY or GYROZ',required=True)

arguments = parser.parse_args()
args = vars(parser.parse_args())

target=arguments.target
print(target)

path='/home/felipeadachi/Dados/Kfolds_cpu/Whole_train'
rf_results=str(path)+'/relieff/'+str(target)
final_set=path+'/relieff/final_set_'+str(target)+'.txt'    


def main():
    correlation_results='/home/felipeadachi/Dados/Kfolds_cpu/Whole_train/mutinfo_analysis/'+str(target)+'/mutinfo_'+str(target)+'.csv'
    
    
    db_json='/home/felipeadachi/Dados'

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
    #print("dft>",dft)
    print("into R>")
    r=pr.R(use_pandas=True) 
    r.assign("rdf2", dft)
    r.assign("target",target)
    r('library(CORElearn)')            
    r('cat(versionCore(),"\n")')
    r('estReliefF <- attrEval(target, rdf2,estimator="ReliefFexpRank", ReliefIterations=0)')
    
    estReliefF=r.estReliefF
    with open(rf_results,'a') as f:
        f.write('estimator="ReliefFexpRank", ReliefIterations=0. Neighbours=70, sigma=20\n')
        f.write(str(list(df))+"\n")
        f.write(str(estReliefF)+"\n")
    print("End of process")
    
    write_final_results(rf_results,final_set)
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

def findnth(string, substring, n):
    parts = string.split(substring, n + 1)
    if len(parts) <= n + 1:
        return -1
    return len(string) - len(parts[-1]) - len(substring)

#---------------------#---------------------#---------------------#---------------------#---------------------#

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

#---------------------#---------------------#---------------------#---------------------#---------------------#

def write_final_results(filename,path_out):
    with open(filename,'r') as f:
        s=f.read()
    
    feature=s[s.find("[")+1:s.find("]")]
    
    
    weight=s[findnth(s,"[",1):findnth(s,"]",1)]
    
    weight=re.sub("\s+", ",", weight.strip())
    weight=weight[2:]
    weights_pre=[]
    weights=[]
    features=[]
    
    for ft in feature.split(','):
        #print(float(wt))
        features.append(ft)
    
    #print(features)
    
    for wt in weight.split(','):
        #print(float(wt))
        weights_pre.append(float(wt))
    #print(weights)
    
    
    
    
    weights=sorted(weights_pre,reverse=True)
    #weights = [float(i)/max(weights0) for i in weights0]
    print("weights>",weights)
    ratio=[]
    diff=[]
    i=0
    #print(weights[1])
    for wt in weights:
        if i != len(weights)-1:
            ratio.append(weights[i]/weights[i+1])
            #print(ratio[i])
            if i > 0:
                diff.append(ratio[i]-ratio[i-1])
        i+=1
    print("ratio>",ratio)
    print("diff>",diff)
    diff[0]=0
    diff[-1]=0
    print("diff2>",diff)
    print("max>",max(diff,key=abs))
    print("index>",diff.index(max(diff,key=abs)))
    with open(filename+'_log.txt','a') as f:
        print("inclusive threshold>",weights[diff.index(max(diff,key=abs))],file=f)
        print("\n",file=f)
    print("inclusive threshold>",weights[diff.index(max(diff,key=abs))])
    #print(len(weights),len(ratio),len(diff))
    thresh=weights[diff.index(max(diff,key=abs))]
    final_features=[]
    
    for x,y in zip(features,weights_pre):
            if float(y)>=float(thresh):
                final_features.append(x)
                print(x,y)
                with open(filename+'_log.txt','a') as f:
                    print(x,y,file=f)
            else:
                print(x,y,"não selecionado")
                with open(filename+'_log.txt','a') as f:
                    print(x,y,"não selectionado",file=f)
    with open(path_out,'a') as f:
        print(final_features,file=f)
    


#---------------------#---------------------#---------------------#---------------------#---------------------#

def get_corr_results(path):
    
    data=pd.read_csv(path)
    which_topast = []

    for i in data.index:
        if str(data.loc[i][0]) != 'deapth' and str(data.loc[i][0]) != 'temp':
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
