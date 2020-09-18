#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 17:16:19 2019

@author: felipeadachi
"""

#Ref: doi:10.1016/j.knosys.2011.08.021 - How to select cutoff
#doi:10.1016/j.ejor.2006.09.103 - Why eliminate first and last weight
import re
import os
def findnth(string, substring, n):
    parts = string.split(substring, n + 1)
    if len(parts) <= n + 1:
        return -1
    return len(string) - len(parts[-1]) - len(substring)

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file
print("hello")
pth='/home/felipeadachi/Dados/Kfolds_cpu/Feature_Selection/Relief/'

pth2='Results_Relief/GYROY/'
for filename in files(pth+pth2):
    print("in")
    with open(pth+pth2+filename,'r') as f:
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
    
    
    
    
    #-------------------#----------------------#---------------------
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
    with open(pth+'Final_set/GYROY/'+filename,'a') as f:
        print("inclusive threshold>",weights[diff.index(max(diff,key=abs))],file=f)
        print("\n",file=f)
    print("inclusive threshold>",weights[diff.index(max(diff,key=abs))])
    #print(len(weights),len(ratio),len(diff))
    thresh=weights[diff.index(max(diff,key=abs))]
    for x,y in zip(features,weights_pre):
            if float(y)>=float(thresh):
                print(x,y)
                with open(pth+'Final_set/GYROY/'+filename,'a') as f:
                    print(x,y,file=f)
            else:
                print(x,y,"não selecionado")
                with open(pth+'Final_set/GYROY/'+filename,'a') as f:
                    print(x,y,"não selectionado",file=f)
