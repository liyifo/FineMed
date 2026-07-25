import dill

import pandas as pd


data = dill.load(open('./data_final.pkl', "rb"))
#data = pd.read_csv('./latest_records.csv')
def transICD(code, is_diag):
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    
    return code


for index, row in data.iterrows():
    print(f'{index}/{len(data)}')
    diag = row['ICD9_CODE']
    proc = row['PRO_CODE']
    text = row['TEXT']

    for i in range(len(diag)):
        diag[i] = transICD(diag[i], True)

    for i in range(len(proc)):
        proc[i] = transICD(proc[i], False)

    data.at[index, 'ICD9_CODE'] = diag
    data.at[index, 'PRO_CODE'] = proc
    

data.to_pickle('data_final.pkl')