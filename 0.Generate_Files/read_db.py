#Este script gera arquivos em JSON, armazenados em um dado pickle, a partir 
#de um path para os arquivos coletados do ROV. É possível usar a variável keyword
#para gerar arquivos apenas de dados referentes a sessões específicas, como velocidade ou manobra.


import json
import os
import collections
import pickle
from attributes import past_to_get,which_topast
import numpy as np

#---------------------#---------------------#---------------------#---------------------#---------------------#    


def read_json_content(filename):
    with open(filename, 'r') as f:
        return json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(f.read(os.stat(filename).st_size))

#---------------------#---------------------#---------------------#---------------------#---------------------#


path='/home/felipeadachi/Dados/DB-Test'
keyword=''

def main():
    #df=pd.DataFrame()
    js_prev=[]
    filenames=([os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(path)) for f in fn])
    for filename in filenames:
        if keyword in filename:
            js=read_json_content(filename)
            js_prev=js_prev+js['0']['data']
            #js=past_commands(js)
            #print(js)
            #df=df.append(pd.DataFrame(js['0']['data']))
    
    print("js>",js_prev[0])
    with open('DB-Test', 'wb') as f:
        pickle.dump(js_prev,f)
            
    #for arquivo in sorted(os.listdir(path)):
    #    js=read_json_content(arquivo)
    
    
    #print(type(js['0']['data']))
    
    #df=pd.DataFrame(js['0']['data'])
    #df=df.set_index('timestamp')
    #print("df>",df)
    
#---------------------#---------------------#---------------------#---------------------#---------------------#

def past_commands(content):
    
    data_list=content['0']['data']
    
    i=0
    
        
    while (i<len(data_list)):
        
        #get_token retorna o número de vezes permitido para pegar os comandos passados, com base nos flags "has_elapsed" de cada timestep
        token=get_token(i,data_list)
        #print("token for timestep:",i,"is:",token)
        data=data_list[i]
        tkn={}
        
        for att in which_topast:
                    tkn[att[0]]=token
                
        for past in past_to_get:
            
            if i<past:
                for att in which_topast:
                    data[att[0]+'_t'+str(past)]=np.nan

            else:
                    
                for att in which_topast:
                    
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
    
    return content

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



if (__name__=='__main__'):
    main()
else:main()