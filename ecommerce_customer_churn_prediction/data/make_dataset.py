import pandas as pd

def load_data(file_path, kind='csv'):
    
    if kind not in ['csv', 'xlsx']:
        raise ValueError('The argument kind can only be "csv" or "xlsx"')
    
    if kind == 'xlsx':
        data = pd.read_excel(file_path, sheet_name=1)
    else:
        data = pd.read_csv(file_path)
    
    return data