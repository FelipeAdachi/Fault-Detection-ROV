#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:23:51 2019

@author: felipeadachi
"""

import random
import copy
import numpy as np

rand=random.Random()
#rand.seed(2)

def inject_fault(source_series_0,scenario):
    
    scenario_number=scenario['scenario_number']
    duration=int(max(scenario['duration_fault']))
    when_active=scenario['insert_when_active']
    number_faults=scenario['number_faults']
    gap=scenario['gap_between_faults']
    source_series=copy.deepcopy(source_series_0)
    create_fault_feature(source_series)
    #print("dur>",duration)
    #ignore first and last n timesteps of a run when applying faults.
    #(each run is considered to be contiguous time intervals with has_elapsed==0)
    ignore_first=scenario.get('ignore_first',25)
    ignore_last=scenario.get('ignore_last',25)
        
    #print(valid_indexes)
    valid_indexes=check_elapsed(source_series,duration,when_active,ignore_first,ignore_last)
    #print("initial valid indexes>",valid_indexes)
    for fault in range(number_faults):
        print("fault number:",fault,"for scenario",scenario_number,"\n")
        #print("initial valid indexes",valid_indexes)
        duration=rand.choice(scenario['duration_fault'])
        offset = rand.choice(valid_indexes)
        #print("offset>",offset)
        valid_indexes=update_valid_indexes(valid_indexes,offset,duration,gap)
        #print("updated valid indexes>",valid_indexes)
        #input("Press Enter to continue...")

        if not valid_indexes:
            print("list is empty")
            source_series=None
            return source_series
            
        #print(valid_indexes)
        source_series=mark_faults(source_series,offset,duration)
        #print(scenario['stuck_at']['active'])
        
        apply_stuck_at(scenario,source_series,offset,duration)
        apply_constant_gain(scenario,source_series,offset,duration)
        apply_constant_bias(scenario,source_series,offset,duration)
        apply_drift(scenario,source_series,offset,duration)
        print("ignore first>",ignore_first)
        #print(offset)
        #print(source_series[offset:offset+int(duration)])
    return source_series
    
#-------------#---------------------#------------------------#----------------
    
    
def check_elapsed(source_series,duration,when_active,ignore_first,ignore_last):
    all_indexes=list(range(len(source_series)-duration))
    
    nonvalid_indexes=[]
    nonvalid=False
    #Ensures that fault is injected in contiguous time intervals
    for i in range(ignore_first,len(source_series)-duration-ignore_last+1):
        for j in range(0,duration):
            if source_series[i+j]['has_elapsed']:
                nonvalid=True
        if nonvalid:
            nonvalid_indexes.append(i)
        nonvalid=False
    valid_indexes= [x for x in all_indexes if x not in nonvalid_indexes]
    
    nonvalid_indexes=[]
    nonvalid=False
    #Insert faults only when commands are sent to motors
    if when_active=='True':
        for i in range(0,len(source_series)-duration):
            for j in range(0,duration):
                if source_series[i+j]['mtarg1']==1500 and source_series[i+j]['mtarg3']==1500:
                    nonvalid=True
            if nonvalid:
                nonvalid_indexes.append(i)
            nonvalid=False
        valid_indexes= [x for x in valid_indexes if x not in nonvalid_indexes]
                
    
    return valid_indexes

#-------------#---------------------#------------------------#----------------

def mark_faults(source_series,offset,duration):
    for i in range(duration):
        source_series[offset+i]['fault']=0

    return source_series

#-------------#---------------------#------------------------#----------------

def create_fault_feature(series):
    for i in range(len(series)):
        series[i]['fault']=np.nan
        
#-------------#---------------------#------------------------#----------------
        
def apply_stuck_at(scenario,source_series,offset,duration):
            
            if scenario['stuck_at']['active']=="True":
                
                feature=scenario['stuck_at']['feature']
                feature=feature[0]
                constant=rand.choice(scenario['stuck_at']['constant'])
                print("Stuck-at Constant>",constant)
                print("Duration>",duration)
                if constant=='last':
                    #constant variable gets replaced by the last value without fault. the stuck-at fault will mantain this value for the duration of the fault
                    #print("last value in if>",source_series[offset][feature])
                    constant=source_series[offset][feature]
                
                for i in range(duration):
                    #print("last value outside if>",source_series[offset][feature])
                    source_series[offset+i][feature]=constant
                    source_series[offset+i]['fault_type']='stuck-at'
                    source_series[offset+i]['fault_value']=constant
                    source_series[offset+i]['fault_duration']=duration
                #print("hello")


#-------------#---------------------#------------------------#----------------

def apply_constant_gain(scenario,source_series,offset,duration):
    if scenario['constant_gain']['active']=="True":
        features=scenario['constant_gain']['feature']
        feature=features[0]
        coefficient=rand.choice(scenario['constant_gain']['coefficient'])
        for i in range(duration):
            source_series[offset+i][feature]=coefficient*float(source_series[offset+i][feature])
            source_series[offset+i]['fault_type']='constant_gain'
            source_series[offset+i]['fault_value']=coefficient
            source_series[offset+i]['fault_duration']=duration
        #print("hello")

#-------------#---------------------#------------------------#----------------

def apply_constant_bias(scenario,source_series,offset,duration):
    if scenario['constant_bias']['active']=="True":
        features=scenario['constant_bias']['feature']
        #this line is a design change: each scenario now admits only one feature. simultaneous faults are no longer being considered
        feature=features[0]
        constant=rand.choice(scenario['constant_bias']['constant'])
        for i in range(duration):
            source_series[offset+i][feature]=float(source_series[offset+i][feature])+float(constant)
            source_series[offset+i]['fault_type']='constant_bias'
            source_series[offset+i]['fault_value']=constant
            source_series[offset+i]['fault_duration']=duration
#-------------#---------------------#------------------------#----------------
        
def apply_drift(scenario,source_series,offset,duration):
    if scenario['drift']['active']=="True":
        features=scenario['drift']['feature']
        feature=features[0]
        slope=rand.choice(scenario['drift']['slope'])
        for i in range(duration):
            source_series[offset+i][feature]=float(source_series[offset+i][feature])+float(slope*(i+1))
            source_series[offset+i]['fault_type']='drift'
            source_series[offset+i]['fault_value']=slope
            source_series[offset+i]['fault_duration']=duration
#-------------#---------------------#------------------------#----------------

def update_valid_indexes(valid_indexes,offset,duration,gap):
    
    to_remove=[None]*(2*duration+2*gap)
    j=0
    for i in range(offset-gap-duration,offset+duration+gap):
        to_remove[j]=i
        j+=1
    #print("offset is>",offset,"fault_duration is>",duration,"gap is>",gap)
    #print("to remove>",to_remove)
    valid_indexes2= [x for x in valid_indexes if x not in to_remove]
    return valid_indexes2