#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:46:54 2019

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


    

def main():
    
    for k in range(1,6):
        path='/home/felipeadachi/Dados/Kfolds_cpu/Fold_'+str(k)
        file=path+'/Train_'+str(k)
        df=pd.DataFrame()
        aux=[]
        js=read_json_content(file)

        which_topast='which_topast_'+str(k)
        for att in eval(which_topast):
            js2=past_commands(js,att)
            att2=att[0]+'_t'+str(att[1])
            aux.append(att2)
        which_tokeep=['GYROZ']
        which_tokeep=which_tokeep+aux
        #print(which_tokeep)
        
        df=df.append(pd.DataFrame(js2))
        
        for col in df.columns:
                if col not in which_tokeep:
                    df=df.drop(columns=col)
        
        df=df.dropna()
        
        y=df['GYROZ'].values
        #print(y.shape)
        y=y.reshape(-1,1)
        
        min_max_scaler_y = preprocessing.MinMaxScaler()
        y = min_max_scaler_y.fit_transform(y)
        y=y.ravel()
        #print("x",len(x))
        df=df.drop(columns='GYROZ')
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
        with open(path+'/stepwise/GYROZ.txt','a') as f:
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
            with open(path+'/stepwise/GYROZ.txt','a') as f:
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
            with open(path+'/stepwise/GYROZ.txt','a') as f:
                print("Feature list after removal:\n",file=f)
                print(listdf,"\n",file=f)
            nstep+=1
            
        #print("final vifs>",vif)
        
        regressor_OLS = sm.OLS(endog = y, exog = x).fit()
        with open(path+'/stepwise/GYROZ_OLS_Step_0.txt','a') as f:
            print(regressor_OLS.summary(),file=f)
        
        with open(path+'/stepwise/GYROZ_OLS_Step_0_Latex.txt','a') as f:
            print(regressor_OLS.summary().as_latex(),file=f)
        
        
        
        with open(path+'/stepwise/GYROZ.txt','a') as f:
            print("#------------#------------#---------------#\n")
            print("Initiating OLSRegression for p-value removal (p-value>0.01)\n",
                  file=f)
            print("Initial pvalues for Step 0:",regressor_OLS.pvalues,
                  file=f)
        nstep=1
        while max(regressor_OLS.pvalues[1:])>0.01:
            idx=int(np.where(regressor_OLS.pvalues==max(regressor_OLS.pvalues[1:]))[0])
            with open(path+'/stepwise/GYROZ.txt','a') as f:
                print("STEP",nstep,"\n",file=f)
                print("Highest p-value of",max(regressor_OLS.pvalues[1:]),
                      "for feature",listdf[idx-1],file=f)
            listdf.pop(idx-1)        
            x=np.delete(x,[idx],1)
            regressor_OLS = sm.OLS(endog = y, exog = x).fit()
            with open(path+'/stepwise/GYROZ_OLS_Step_'+str(nstep)+'.txt','a') as f:
                print(regressor_OLS.summary(),file=f)
        
            with open(path+'/stepwise/GYROZ_OLS_Step_'+str(nstep)+'_Latex.txt','a') as f:
                print(regressor_OLS.summary().as_latex(),file=f)
            nstep+=1
            with open(path+'/stepwise/GYROZ.txt','a') as f:
                print("Features after removal>",listdf,file=f)
                
        print("KFold",k,">\n")
        print(listdf)
        #print("vif>",vif)
        #print(x[14])
        #print(VIF)
        
        #print(max(VIF.items(), key=operator.itemgetter(1))[0])
        
        
        
        #print("xshape",x.shape)
    
        #print(js2[1])
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
    ["mtarg1",2],
    ["mtarg2",2],
    ["mtarg3",2],
    #["deapth",0],
    ["roll",1],
    ["pitch",1],
    ["LACCX",1],
    ["LACCY",1],
    ["LACCZ",1],
    ["GYROX",2],
    ["GYROY",1],
    ["SC1I",2],
    ["SC2I",2],
    ["SC3I",2],
    ["BT1I",2],
    ["BT2I",2],
    ["vout",1],
    ["iout",1],
    ["cpuUsage",1]
    #["temp",0]
    
    ]

which_topast_2 = [
    ["mtarg1",2],
    ["mtarg2",2],
    ["mtarg3",2],
    #["deapth",0],
    ["roll",1],
    ["pitch",1],
    ["LACCX",1],
    ["LACCY",1],
    ["LACCZ",1],
    ["GYROX",4],
    ["GYROY",1],
    ["SC1I",2],
    ["SC2I",3],
    ["SC3I",2],
    ["BT1I",2],
    ["BT2I",2],
    ["vout",1],
    ["iout",1],
    ["cpuUsage",1]
    #["temp",0]
    
    ]

which_topast_3 = [
    ["mtarg1",2],
    ["mtarg2",2],
    ["mtarg3",2],
    #["deapth",0],
    ["roll",1],
    ["pitch",1],
    ["LACCX",1],
    ["LACCY",1],
    ["LACCZ",1],
    ["GYROX",2],
    ["GYROY",1],
    ["SC1I",2],
    ["SC2I",1],
    ["SC3I",2],
    ["BT1I",2],
    ["BT2I",2],
    ["vout",1],
    ["iout",1],
    ["cpuUsage",1]
    #["temp",0]
    
    ]

which_topast_4 = [
    ["mtarg1",2],
    ["mtarg2",2],
    ["mtarg3",2],
    #["deapth",0],
    ["roll",1],
    ["pitch",1],
    ["LACCX",1],
    ["LACCY",1],
    ["LACCZ",1],
    ["GYROX",2],
    ["GYROY",1],
    ["SC1I",2],
    ["SC2I",2],
    ["SC3I",2],
    ["BT1I",2],
    ["BT2I",2],
    ["vout",1],
    ["iout",1],
    ["cpuUsage",1]
    #["temp",0]
    
    ]

which_topast_5 = [
    ["mtarg1",2],
    ["mtarg2",2],
    ["mtarg3",2],
    #["deapth",0],
    ["roll",1],
    ["pitch",1],
    ["LACCX",1],
    ["LACCY",1],
    ["LACCZ",1],
    ["GYROX",4],
    ["GYROY",1],
    ["SC1I",2],
    ["SC2I",2],
    ["SC3I",2],
    ["BT1I",2],
    ["BT2I",2],
    ["vout",1],
    ["iout",1],
    ["cpuUsage",1]
    #["temp",0]
    
    ]





if (__name__=='__main__'):
    main()