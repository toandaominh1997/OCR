import pandas as pd 
import os 
import subprocess
import copy
import datetime
import torch
def get_vocab(root, label):
    # filename = os.path.join(root, label)

    # df = pd.read_json(filename, typ='series', encoding="utf-8")
    # df = pd.DataFrame(df)
    # df = df.reset_index()
    # df.columns = ['index', 'label']
    # alphabets = ''.join(sorted(set(''.join(df['label'].get_values()))))

    typefile = label.split('.')[-1]
    list_label = list()
    if(typefile=='json'):
        df = pd.read_json(os.path.join(root, label), typ='series')
        df = pd.DataFrame(df)
        df = df.reset_index()
        df.columns = ['index', 'label']
        list_label = df['label'].tolist()
    elif(typefile=='txt'):
        lines = list(open(os.path.join(root, label)))
        for line in lines:
            label = line.split('|')[1].replace('\n', '')
            list_label.append(label)
    
    alphabets = ''.join(sorted(set(''.join(list_label))))
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

def save_checkpoint(epoch, model, optimizer, save_dir, save_best, start_time):
    
    checkpoint_dir = os.path.join(save_dir, start_time)
    arch = type(model).__name__
    model = copy.deepcopy(model)
    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    filename = os.path.join(checkpoint_dir, 'checkpoint_epoch{}.pth'.format(epoch))
    torch.save(state, filename)
    print('Saving checkpoint: {} ...'.format(filename))
    if(save_best):
        best_path = os.path.join(checkpoint_dir, 'model_best.pth')
        torch.save(state, best_path)
        print('Saving current best: {} ...'.format('model_best.pth'))

def resume_checkpoint(model, optimizer, resume_path):
    checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer