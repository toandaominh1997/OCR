import pandas as pd 
import os 
def get_vocab(root, label):
    filename = os.path.join(root, label)
    df = pd.read_json(filename, typ='series', encoding="utf-8")
    df = pd.DataFrame(df)
    df = df.reset_index()
    df.columns = ['index', 'label']
    alphabets = ''.join(sorted(set(''.join(df['label'].get_values()))))
    return alphabets