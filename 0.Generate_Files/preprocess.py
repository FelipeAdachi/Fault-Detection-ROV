#!/usr/bin/python3

#Este programa desacumula os jsons, isto é: compara um arquivo com o subsequente e deleta qualquer timestamp menor do que o timestamp máximo encontrado no arquivo anterior, tanto para os dados de navegação quanto para o de telemetria. Deve-se especificar o endereço dos dados acumulados de entrada e o endereço dos dados desacumulados de saída. Deve-se também especificar uma data de início. No primeiro arquivo, começando com "0.", o programa irá deletar qualquer dado anterior à 00:00 da data especificada.

#Exemplo: python3 preprocess.py -i /home/felipeadachi/Dados/cumulating/cumulating2-13.03.19 -o /home/felipeadachi/Dados/non-cumulating/13.03.19 -d 2019,3,13


import json
import os
import collections
import datetime
import argparse
import numpy as np
from attributes import which_topast,past_to_get

input_1_filename = 'nonein.json'
output_filename = 'noneout.json'


parser = argparse.ArgumentParser(description='Preprocess Raw JSON files')

required = parser.add_argument_group('required named arguments')

required.add_argument('-i','--input', help='Location of local data to create database from', required=True)
required.add_argument('-o','--output', help='Output path', required=True)
required.add_argument('-d','--date', help='Date to decumulate', required=True)




args = vars(parser.parse_args())


date=args['date'].split(',')
folder_path_in = args['input']
folder_path_out = args['output']

which_todelta = [['deapth']]



#                                    Y   M  D
date_to_start=int(datetime.datetime(2019,1,19).strftime('%s'))*1000


date_to_start=int(datetime.datetime(int(date[0]),int(date[1]),int(date[2])).strftime('%s'))*1000


#---------------------#---------------------#---------------------#---------------------#---------------------#

def main():

    i=0
    number_of_files = len(os.listdir(folder_path_in))
    list_of_files = [None]*number_of_files
    print(number_of_files)
    for data_file in sorted(os.listdir(folder_path_in)):
          
        numfile=data_file.partition(".")[0]
        print(numfile)
        list_of_files[int(numfile)]=data_file
        print(numfile+">"+data_file)
    
    print(list_of_files)

    while i < number_of_files:
        json_current_content= read_json_content(folder_path_in + '/' + list_of_files[i])
        if (i != 0):
            json_preceding_content = read_json_content(folder_path_in + '/' + list_of_files[i-1])
            json_file_content = (json_preceding_content,json_current_content)
            print("max timestamps:")
            json_content_out = content_modifier(*json_file_content)
        else:
            json_content_out= check_start_date(json_current_content)
        
        json_content_out=motorsep(json_content_out)
        
        json_content_out=has_elapsed(json_content_out)
            
        json_content_out = delta(json_content_out)
        
        #json_content_out = past_commands(json_content_out,which_topast)
        
        
        #del json_content_out['0']
        if not os.path.exists(folder_path_out):
            os.makedirs(folder_path_out)
        write_json_content(folder_path_out + '/' + list_of_files[i],json_content_out)
        i += 1 
        
        
        
        
        #json_file_content=read_json_content(folder_path_in+'/'+data_file)
        #del json_file_content['0']
    
        #write_json_content(folder_path_out+'/'+data_file,json_file_content)

        #json_file_content = None
    
        #json_file_content = read_json_content(folder_path_in+'/'+data_file)
        
        #if json_file_content is None:
            #exit(0)
        #print(data_file)
        #output_filename=folder_path_out+'/'+data_file
        #write_json_content(output_filename, json_file_content)

#---------------------#---------------------#---------------------#---------------------#---------------------#

def check_start_date(current_content):
    
    for i in range(0,2):
        
        data_list = current_content[str(i)]['data']
        
        l=0
        
        while l < len(data_list):
            
            data = data_list[l]
            cur_timestamp = data['timestamp']
            
            if cur_timestamp < date_to_start:
                #print("current timestamp:",cur_timestamp,"type:",type(cur_timestamp))
                #print("date_to_start:",date_to_start,"type:",type(date_to_start))
                
                del data_list[l]
                
            else:
                l=l+1
        

        
    return current_content

#---------------------#---------------------#---------------------#---------------------#---------------------#

def content_modifier(content_1, content_2):
    
    timestamps_nav = list()
    timestamps_telemetry = list()
    
    for data_nav in content_1['0']['data']:
        timestamps_nav.append(data_nav['timestamp'])
    
    max_timestamp_nav_preceding = max(timestamps_nav)
    print(max_timestamp_nav_preceding)
    data_list_nav = content_2['0']['data']
    
    l = 0
    
    while l < len(data_list_nav):
            
        data_nav = data_list_nav[l]
        cur_timestamp_nav = data_nav['timestamp']
        
        if cur_timestamp_nav <= max_timestamp_nav_preceding:
            del data_list_nav[l]
            
        else:
            l = l + 1
            
    for data_telemetry in content_1['1']['data']:
        timestamps_telemetry.append(data_telemetry['timestamp'])
    
    max_timestamp_telemetry_preceding = max(timestamps_telemetry)
    print(max_timestamp_telemetry_preceding)
    data_list_telemetry = content_2['1']['data']
    
    l = 0
    
    while l < len(data_list_telemetry):
        
        data_telemetry = data_list_telemetry[l]
        cur_timestamp_telemetry = data_telemetry['timestamp']
        
        if cur_timestamp_telemetry <= max_timestamp_telemetry_preceding:
            del data_list_telemetry[l]
            
        else:
            l = l + 1
    
    return content_2

#---------------------#---------------------#---------------------#---------------------#---------------------#

def motorsep(content_1):
    

    data_list = content_1['0']['data']
    
    i = 0
    while i < len(data_list):
        #print(i)
        data = data_list[i]
        nomotors=False
        nomtargs=False
        #try:
        #    motor1,motor2,motor3 = (data['motors'].split(","))
        
        #except KeyError:
        #    nomotors=True

        
        #if nomotors==False:
        #    data['motor1']=int(motor1)
        #    data['motor2']=int(motor2)
        #    data['motor3']=int(motor3)

        
        try:    
            mtarg1,mtarg2,mtarg3 = data['mtarg'].split(",")
        except KeyError:
            print("key error mtarg:",data['timestamp'])
            nomtargs=True

        if nomtargs==False:
            data['mtarg1']=int(mtarg1)
            data['mtarg2']=int(mtarg2)
            data['mtarg3']=int(mtarg3)
        
        i += 1
    
    return content_1


#---------------------#---------------------#---------------------#---------------------#---------------------#

def read_json_content(filename):
    with open(filename, 'r') as f:
        return json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(f.read(os.stat(filename).st_size))

#---------------------#---------------------#---------------------#---------------------#---------------------#
def delta(content):
    
    data_list=content['0']['data']
    
    i=0
    
    while (i<len(data_list)):
        
        data=data_list[i]
        for attribute in which_todelta:
            if i==0 or data['has_elapsed']==1:
                data[attribute[0]+'_delta'] = np.nan
                try:
                    prev=data[attribute[0]]
                except KeyError:
                    print("key error:",attribute)
            else:
                data_prev=data_list[i-1]
                if(attribute==["yaw"]):
                    #print("aqui")
                    #Quando o rov passa de 340 para 10 graus por exemplo, o delta seria calculado como 330 graus, quando na verdade ele deslocou apenas 20 graus
                    if abs(float(data[attribute[0]])-float(data_prev[attribute[0]]))>360-abs(float(data[attribute[0]])-float(data_prev[attribute[0]])):
                        data[attribute[0]+'_delta'] = float("%.2f" % round(360-(abs(float(data[attribute[0]])-float(data_prev[attribute[0]]))),2))
                        #print("aqui:",data[attribute[0]],"timestamp:",data['timestamp'])
                        
                    else:
                        data[attribute[0]+'_delta'] = float("%.2f" % round(float(data[attribute[0]])-float(data_prev[attribute[0]]),2))
                        
                else:
                    try:
                        if data[attribute[0]]==data_prev[attribute[0]]:
                            data[attribute[0]+'_delta'] = data_prev[attribute[0]+'_delta']
                        else:
                            data[attribute[0]+'_delta'] = float("%.2f" % round((float(data[attribute[0]])-float(data_prev[attribute[0]])),2))
                
                    except KeyError:
                        print("key error:",attribute)
        i+=1
        
    return content

#---------------------#---------------------#---------------------#---------------------#---------------------#

def has_elapsed(content):
    
    data_list = content['0']['data']
    
    i=0
    
    while (i<len(data_list)):
        
        data=data_list[i]
        if i==0:
            data['has_elapsed']=1
        if i >= 1:
            if (data['timestamp']-data_list[i-1]['timestamp'])>=500:
                data['has_elapsed']=1
            else:
                data['has_elapsed']=0
        i+=1
    return content

#---------------------#---------------------#---------------------#---------------------#---------------------#

def past_commands(content,which_topast):
    
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




def write_json_content(filename, content):
    with open(filename, 'w') as f:
        f.write(json.dumps(content,indent=4))

#---------------------#---------------------#---------------------#---------------------#---------------------#

if(__name__ == '__main__'):
    main();
