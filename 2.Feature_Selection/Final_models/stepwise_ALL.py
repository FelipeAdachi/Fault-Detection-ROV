#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:38:50 2019

@author: felipeadachi
"""


import numpy as np
import pandas as pd
from sklearn import preprocessing
import statsmodels.formula.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import operator
import json
import collections
import os
import pickle
import argparse

path='/home/felipeadachi/Dados/Kfolds_cpu/Whole_train'


parser = argparse.ArgumentParser(description="Stepwise selection, using timeshifted values given by correlation analysis.")
required = parser.add_argument_group('required named arguments')

required.add_argument('-t','--target',choices=['GYROX','GYROY','GYROZ'], help='Target feature. Either GYROX,GYROY or GYROZ',required=True)

arguments = parser.parse_args()
args = vars(parser.parse_args())

target=arguments.target
print(target)

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
    x = np.append ( arr = np.ones([len(x),1]).astype(int), values = x, axis = 1)
    #print("xxxx>",x)
    #regressor_OLS = sm.OLS(endog = y, exog = x).fit()
    #print(regressor_OLS.summary())
    vif = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
    print("vif>",vif)
    listdf=list(df)
    nstep=1
    
    with open(path+'/stepwise/'+str(target)+'.txt','a') as f:
        print("Initiating VIF removal procedure (VIF>=5)\n",file=f)
        print("Starting VIF:",vif,"\n",file=f)
    while max(vif[1:])>=5:
        #print("maxvif>",max(vif))
        #print("dentro while")
        vif.pop(0)
        VIF={}
        for att,vf in zip(listdf,vif):
            #print(att,vf)
            VIF[att]=vf
        with open(path+'/stepwise/'+str(target)+'.txt','a') as f:
            print("Initial VIF for Step",nstep,":\n",file=f)
            print(VIF,"\n",file=f)
            print("Highest VIF of",max(vif),"for feature",
                  max(VIF.items(),key=operator.itemgetter(1))[0],"\n",file=f)
        #Get index of feature whith highest VIF
        index=listdf.index(max(VIF.items(), key=operator.itemgetter(1))[0])
        #index+1 to account for appended row of 1s to calculate VIF
        x=np.delete(x,[index+1],1)
        listdf.pop(index)        
        vif = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
        with open(path+'/stepwise/'+str(target)+'.txt','a') as f:
            print("Feature list after removal:\n",file=f)
            print(listdf,"\n",file=f)
        nstep+=1
        
    #print("final vifs>",vif)
    
    regressor_OLS = sm.OLS(endog = y, exog = x).fit()
    with open(path+'/stepwise/'+str(target)+'_OLS_Step_0.txt','a') as f:
        print(regressor_OLS.summary(),file=f)
    
    with open(path+'/stepwise/'+str(target)+'_OLS_Step_0_Latex.txt','a') as f:
        print(regressor_OLS.summary().as_latex(),file=f)
    
    
    
    with open(path+'/stepwise/'+str(target)+'.txt','a') as f:
        print("#------------#------------#---------------#\n")
        print("Initiating OLSRegression for p-value removal (p-value>0.01)\n",
              file=f)
        print("Initial pvalues for Step 0:",regressor_OLS.pvalues,
              file=f)
    nstep=1
    while max(regressor_OLS.pvalues[1:])>0.01:
        idx=int(np.where(regressor_OLS.pvalues==max(regressor_OLS.pvalues[1:]))[0])
        with open(path+'/stepwise/'+str(target)+'.txt','a') as f:
            print("STEP",nstep,"\n",file=f)
            print("Highest p-value of",max(regressor_OLS.pvalues[1:]),
                  "for feature",listdf[idx-1],file=f)
        listdf.pop(idx-1)        
        x=np.delete(x,[idx],1)
        regressor_OLS = sm.OLS(endog = y, exog = x).fit()
        with open(path+'/stepwise/'+str(target)+'_OLS_Step_'+str(nstep)+'.txt','a') as f:
            print(regressor_OLS.summary(),file=f)
        
        with open(path+'/stepwise/'+str(target)+'_OLS_Step_'+str(nstep)+'_Latex.txt','a') as f:
            print(regressor_OLS.summary().as_latex(),file=f)
        nstep+=1
        with open(path+'/stepwise/'+str(target)+'.txt','a') as f:
            print("Features after removal>",listdf,file=f)
            
    print(listdf)
    with open(path+'/stepwise/final_set_'+str(target)+'.txt','a') as f:
        print(listdf,file=f)
    
    #print("vif>",vif)
    #print(x[14])
    #print(VIF)
    
    #print(max(VIF.items(), key=operator.itemgetter(1))[0])
    
    
    
    #print("xshape",x.shape)

    #print(js2[1])

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
    main();
