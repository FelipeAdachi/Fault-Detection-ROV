#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:05:43 2019

@author: felipeadachi
"""

#This script aims to aggregate results obtained on multiple occasions.
#E.g. preliminary results were obtained with durations 1,5,25,50. Afterwards, results for duration 100 were obtained.

import pandas as pd
import matplotlib.pyplot as plt
from read_json import read_json_content
import numpy as np
import statistics

which_fs={
        'rf':'RF_MUTINFO',
        'sw':'SW_MUTINFO',
        'cfs':'CFS_MUTINFO',
        'all':'ALL_MUTINFO'
        }

paths=['/home/felipeadachi/Dados/serverlab/Detection_Results-23.07.19',
       '/home/felipeadachi/Dados/serverlab/Detection_Results_t100',
       '/home/felipeadachi/Dados/serverlab/Detection_Results_t10']

#The config files are in sequence according to the paths array above. 
fault_types={
        'BIAS':{'config':{
                'GYROZ':['scenario_constantbias_GZ.json','sc_bias_t100_GZ.json','sc_bias_t10_GZ.json'],
                'GYROX':'scenario_constantbias_GX.json',
                'GYROY':'scenario_constantbias_GY.json'},
                'coeff':'constant',
                'name':'constant_bias'
                },
        'GAIN':{'config':{
                'GYROZ':['scenario_constantgain_GZ.json','sc_gain_t100_GZ.json','sc_gain_t10_GZ.json'],
                'GYROX':'scenario_constantgain_GX.json',
                'GYROY':'scenario_constantgain_GY.json'},
                'coeff':'coefficient',
                'name':'constant_gain'
                },
        'DRIFT':{'config':{
                'GYROZ':['scenario_drift_GZ.json','sc_drift_t100_GZ.json','sc_drift_t10_GZ.json'],
                'GYROX':'scenario_drift_GX.json',
                'GYROY':'scenario_drift_GY.json'},
                'coeff':'slope',
                'name':'drift'
                },
        'STUCKAT':{'config':{
                'GYROZ':['scenario_stuckat_GZ.json','sc_stuck_t100_GZ.json','sc_stuck_t10_GZ.json'],
                'GYROX':'scenario_stuckat_GX.json',
                'GYROY':'scenario_stuckat_GY.json'},
                'coeff':'constant',
                'name':'stuck_at'
                },

            }


subsets=['cfs','rf','sw','all']

thresholds=[0.06,0.07,0.08,0.09]

sliding_window=[7,10,13]

arrays=[[0.06,0.06,0.06,0.07,0.07,0.07,0.08,0.08,0.08,0.09,0.09,0.09],[7,10,13,7,10,13,7,10,13,7,10,13]]


cols=['positives','negatives','true_positives','false_positives','latencies']
whole_cols=['whole_positives','whole_negatives','whole_true_positives','whole_false_positives','whole_latencies']
    
target='GYROZ'


def main():
    df_lat = pd.DataFrame(index=subsets, columns=arrays)
    df_fscore = pd.DataFrame(index=subsets, columns=arrays)
    df_precision = pd.DataFrame(index=subsets, columns=arrays)
    df_recall = pd.DataFrame(index=subsets, columns=arrays)
    df_mix = pd.DataFrame(index=subsets, columns=arrays)
    
    for fs in subsets:
        whole_positives=0
        whole_negatives={}
        whole_true_positives={}
        whole_false_positives={}
        whole_latencies={}
        whole_latencies_mean={}
        whole_latencies_std={}
        whole_precision={}
        whole_recall={}
        whole_f_score={}
        
        for threshold in thresholds:
            for window_size in sliding_window:
                column = 'mres_%d_thr_%.2f' % (window_size, threshold)
                whole_true_positives[column]=0
                whole_false_positives[column]=0
                whole_negatives[column]=0
                whole_latencies[column]=[]
                whole_precision[column]=0
                whole_recall[column]=0
                whole_f_score[column]=0
               
        for fault_type in fault_types:
            j=0
            for path in paths:
                total_path=path+'/'+target+'/'+which_fs[fs]+'/'+fault_type
                config_path=path+'/'+target+'/'+which_fs[fs]+'/'+fault_type+'/'+fault_types[fault_type]['config'][target][j]
                #print("total path>",config_path)
                config_file=read_json_content(config_path)
                #print(config_file)
                for scenario in config_file['scenarios']:
                    scenario_number=scenario['scenario_number']
                    #ignore first scenarios, with duration of 1 timestep
                    if scenario_number>6:
                        #print("scenario number>",scenario_number)
                        positives=read_json_content(total_path+'/'+str(scenario_number)+'.positives.json')
                        whole_positives=whole_positives+positives
                        for col,whole_col in zip(cols,whole_cols):
                            js=read_json_content(total_path+'/'+str(scenario_number)+'.'+col+'.json')
                            if col != 'positives':
                                for key in js:
                                    if col == 'latencies':
                                        for latency in js[key]:
                                            eval(whole_col)[key].append(latency)
                                        #print(type(eval(whole_col)[key]))
                                        #print(js[key])
                                        
                                    else:
                                        eval(whole_col)[key]=eval(whole_col)[key]+js[key]
                j+=1
        
        for key in whole_true_positives:
            try:
                whole_recall[key]=whole_true_positives[key]/whole_positives
            except ZeroDivisionError:
                whole_recall[key]=np.nan
            try:
                whole_precision[key]=whole_true_positives[key]/(whole_true_positives[key]+whole_false_positives[key])
            except ZeroDivisionError:
                whole_precision[key]=np.nan
            try:
                whole_f_score[key]=2*(whole_precision[key]*whole_recall[key])/(whole_precision[key]+whole_recall[key])
            except ZeroDivisionError:
                whole_f_score[key]=np.nan
                
        for key in whole_latencies:
            print("key latencies>",whole_latencies[key])
            try:
                whole_latencies_mean[key]=statistics.mean(whole_latencies[key])
                #print("mean latencies>",statistics.mean(latencies[key]))
                
            except statistics.StatisticsError:
                whole_latencies_mean[key]=np.nan
            try:
                whole_latencies_std[key]=statistics.pstdev(whole_latencies[key])
                #print("std latencies>",latencies_std[key])
            except statistics.StatisticsError:
                whole_latencies_std[key]=np.nan
        
        #print("whole positives>",whole_positives)
        #print("whole true positives>",whole_true_positives)
        #print("whole precision>",whole_precision)
        #print("whole recall>",whole_recall)
        #print("whole f_score>",whole_f_score)
        print("whole latencies_mean>",whole_latencies_mean)
        print("whole latencies_std>",whole_latencies_std)
        
        for key in whole_f_score:
            x=key.split("_")
            #print(x[1],x[3])
            df_fscore.loc[fs,(str(x[3]),str(x[1]))]=round(whole_f_score[key],2)
        
        for key in whole_precision:
            x=key.split("_")
            #print(x[1],x[3])
            df_precision.loc[fs,(str(x[3]),str(x[1]))]=round(whole_precision[key],2)
        
        for key in whole_recall:
            x=key.split("_")
            #print(x[1],x[3])
            df_recall.loc[fs,(str(x[3]),str(x[1]))]=round(whole_recall[key],2)
    
        
        for key in whole_latencies_mean:
            x=key.split("_")
            df_lat.loc[fs,(str(x[3]),str(x[1]))]=str(int(whole_latencies_mean[key]))+"$\pm$"+str(int(whole_latencies_std[key]))
            #df.loc[fs,(str(x[3]),str(x[1]))]=str(round(whole_latencies_mean[key],1))+'\n'+chr(177)+str(round(whole_latencies_std[key],1))
    df_lat=df_lat.dropna(axis=1)
    df_fscore=df_fscore.dropna(axis=1)
    df_recall=df_recall.dropna(axis=1)
    df_precision=df_precision.dropna(axis=1)
    
    df_precision = df_precision.applymap(str)
    df_recall = df_recall.applymap(str)
    
    df_mix=df_precision+"/"+df_recall
    
    with pd.option_context('display.max_columns',None):
        print(df_mix)
    #print(df)
    df_lat.to_latex('latencies.tex',escape=False)
    #df_fscore.to_latex('f_score.tex')
    #df_precision.to_latex('precision.tex')
    #df_recall.to_latex('recall.tex')
    #df_mix.to_latex('mix.tex')

if (__name__=='__main__'):
    main();