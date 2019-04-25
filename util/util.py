import pandas as pd 
import os 
import subprocess
def get_vocab(root, label):
    filename = os.path.join(root, label)
    df = pd.read_json(filename, typ='series', encoding="utf-8")
    df = pd.DataFrame(df)
    df = df.reset_index()
    df.columns = ['index', 'label']
    alphabets = ''.join(sorted(set(''.join(df['label'].get_values()))))
    return alphabets

def get_gpu():                                                                                          
    df = pd.DataFrame()                                                                                 
    memory_used = subprocess.check_output([                                                             
            'nvidia-smi', '--query-gpu=memory.used',                                                    
            '--format=csv,nounits,noheader'                                                             
        ], encoding='utf-8')                                                                            
    memory_used = [int(x) for x in memory_used.strip().split('\n')]                                     
    df['used'] = memory_used                                                                            
    del memory_used                                                                                     
                                                                                                        
    memory_total = subprocess.check_output([                                                            
            'nvidia-smi', '--query-gpu=memory.total',                                                   
            '--format=csv,nounits,noheader'                                                             
        ], encoding='utf-8')                                                                            
    memory_total = [int(x) for x in memory_total.strip().split('\n')]                                   
    df['total'] = memory_total                                                                          
    del memory_total 

    memory_free = subprocess.check_output([                                                            
            'nvidia-smi', '--query-gpu=memory.free',                                                   
            '--format=csv,nounits,noheader'                                                             
        ], encoding='utf-8')                                                                            
    memory_free = [int(x) for x in memory_free.strip().split('\n')]  
    df['free'] = memory_free
    del memory_free
    df = df.reset_index()                                                                               
    df = df.sort_values(by=['used'])
    df = df.groupby(["used"]).apply(lambda x: x.sort_values(["free"], ascending = False)).reset_index(drop=True)
    result_id = 0                                                                                       
    if(df['used'][0]==0):                                                                               
        print('Yeah, no one used them')                                                                 
        result_id = df['index'][0]                                                                      
    else:                                                                                               
        result_id = df['index'][0]                                                                      
        print('Oh, has anyone used them')
    del df
    return int(result_id)

def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)
