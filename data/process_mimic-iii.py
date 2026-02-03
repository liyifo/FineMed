import pandas as pd
import dill
import numpy as np
from collections import defaultdict


##### process medications #####
# load med data
def med_process(med_file):
    """读取MIMIC原数据文件，保留pid、adm_id、data以及NDC，以DF类型返回"""
    # 读取药物文件，NDC（National Drug Code）以类别类型存储
    med_pd = pd.read_csv(med_file, dtype={'NDC':'category'})

    # drop不用的数据
    med_pd.drop(columns=['ROW_ID','DRUG_TYPE','DRUG_NAME_POE','DRUG_NAME_GENERIC',
                        'FORMULARY_DRUG_CD','PROD_STRENGTH','DOSE_VAL_RX',
                        'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP', 'GSN', 'FORM_UNIT_DISP',
                        'ROUTE','ENDDATE', 'ICUSTAY_ID'], axis=1, inplace=True)
    med_pd.drop(index = med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    # med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')    
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)  # 重置索引，同时drop原索引
    med_pd['DRUG'] = med_pd['DRUG'].str.lower() # 改为小写
    # def filter_first24hour_med(med_pd):
    #     med_pd_new = med_pd.groupby(by=['SUBJECT_ID','HADM_ID']).head(1).reset_index(drop=True)
    #     med_pd_new = med_pd_new.drop(columns=['DRUG', 'NDC'])
    #     med_pd_new = pd.merge(med_pd_new, med_pd, on=['SUBJECT_ID','HADM_ID','STARTDATE'])
    #     return med_pd_new

    # med_pd = filter_first24hour_med(med_pd)
    # temp_key2 = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    # print(f'{len(temp_key2)}')
    # med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)


    print(med_pd.columns)


    drug_list = med_pd['DRUG'].drop_duplicates().tolist()
    print(f'drug _list len : {len(drug_list)}')
    # with open('ndc_list.pkl', 'wb') as f:
    #     dill.dump(drug_list, f)

    return med_pd



def nametoSMILES(druginfo, med_pd):
    drug2smiles = {}
    nametosmiles = {}
    for drugname, smiles in druginfo[["name", "moldb_smiles"]].values:
        if type(smiles) == type("a"):
            drug2smiles[drugname.lower()] = smiles



    for drugname in med_pd["DRUG"].values:
        try:
            nametosmiles[drugname.lower()] = drug2smiles[drugname.lower()]
        except:
            pass

    return nametosmiles



# visit >= 2
def process_visit_lg2(med_pd):
    """筛除admission次数小于两次的患者数据"""
    a = med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
    a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x:len(x))
    a = a[a['HADM_ID_Len'] > 1]
    return a



##### process diagnosis #####
def diag_process(diag_file):
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM','ROW_ID'],inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID','HADM_ID'], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)

    def filter_2000_most_diag(diag_pd):
        diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
        diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:1999, 'ICD9_CODE'])]
        
        return diag_pd.reset_index(drop=True)

    # diag_pd = filter_2000_most_diag(diag_pd)

    return diag_pd

##### process procedure #####
def procedure_process(procedure_file):
    pro_pd = pd.read_csv(procedure_file, dtype={'ICD9_CODE':'category'})
    pro_pd.drop(columns=['ROW_ID'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    pro_pd.drop(columns=['SEQ_NUM'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd


def labtest_process(labevent_file, d_labitem_file, med_pd):
    labevent = pd.read_csv(labevent_file, usecols=['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'VALUEUOM', 'FLAG'])
    d_item = pd.read_csv(d_labitem_file, usecols=['ITEMID', 'LABEL'])
    labevent = labevent.merge(d_item, on=['ITEMID'], how='inner')
    labevent = labevent.dropna(subset=['VALUENUM'])


    labevent['CHARTTIME'] = pd.to_datetime(labevent['CHARTTIME'])
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'])

    med_pd_grouped = med_pd.groupby(['SUBJECT_ID', 'HADM_ID'])['STARTDATE'].agg(['min', 'max']).reset_index()
    print(f'过滤前labtest数量:{len(labevent)}')
    result = []
    for _, med_row in med_pd_grouped.iterrows():
        subject_id = med_row['SUBJECT_ID']
        hadm_id = med_row['HADM_ID']
        min_start_date = med_row['min']
        max_start_date = med_row['max']
        
        # 筛选出相同 SUBJECT_ID 和 HADM_ID 的记录
        sub_labevent = labevent[(labevent['SUBJECT_ID'] == subject_id) & 
                                (labevent['HADM_ID'] == hadm_id)]
        
        # 定义时间范围：最小 start_date 前三天到最大 start_date
        date_range_start = min_start_date - pd.Timedelta(days=3)
        date_range_end = max_start_date + pd.Timedelta(days=1)
        
        # 筛选出在这个时间范围内的记录
        filtered_labevent = sub_labevent[(sub_labevent['CHARTTIME'] >= date_range_start) & 
                                         (sub_labevent['CHARTTIME'] <= date_range_end)]
        
        # 按 ITEMID 分组，保留每个 ITEMID 最早的记录
        earliest_per_itemid = filtered_labevent.sort_values(by='CHARTTIME').groupby('ITEMID').first().reset_index()
        
        result.append(earliest_per_itemid)
    labevent = pd.concat(result, ignore_index=True)
    print(f'过滤后labtest数量:{len(labevent)}')


    def min_max_normalize(x):
        return (x - x.min()) / (x.max() - x.min())

    def z_score_normalize(x):
        return (x - x.mean()) / x.std()
    
    labevent['min_max_norm'] = labevent.groupby('ITEMID')['VALUENUM'].transform(min_max_normalize)
    labevent['z_score_value'] = labevent.groupby('ITEMID')['VALUENUM'].transform(z_score_normalize)

    abnormal_labevent = labevent[labevent['FLAG'] == 'abnormal'].copy()
    abnormal_labevent['test_text'] = abnormal_labevent['LABEL'] + ': ' + abnormal_labevent['VALUENUM'].astype(str) + ' ' + abnormal_labevent['VALUEUOM']
    abnormal_labevent['test_mnorm'] = abnormal_labevent['LABEL'] + ': ' + abnormal_labevent['min_max_norm'].astype(str)
    abnormal_labevent['test_znorm'] = abnormal_labevent['LABEL'] + ': ' + abnormal_labevent['z_score_value'].astype(str)
    abnormal_labevent.dropna(inplace=True)
    abnormal_labevent.drop(columns=['LABEL','ITEMID', 'VALUENUM', 'VALUEUOM', 'FLAG', 'min_max_norm', 'z_score_value'],inplace=True)
    abnormal_labevent.drop_duplicates(inplace=True)
    abnormal_labevent.sort_values(by=['SUBJECT_ID','HADM_ID', 'CHARTTIME'], inplace=True)
    
    # def filter_first24hour_lab(abnormal_labevent):
    #     abnormal_labevent_new = abnormal_labevent.groupby(by=['SUBJECT_ID','HADM_ID']).head(1).reset_index(drop=True)
    #     abnormal_labevent_new = abnormal_labevent_new[['SUBJECT_ID','HADM_ID', 'CHARTTIME']]
    #     abnormal_labevent_new = pd.merge(abnormal_labevent_new, abnormal_labevent, on=['SUBJECT_ID','HADM_ID','CHARTTIME'])
    #     return abnormal_labevent_new
    # abnormal_labevent = filter_first24hour_lab(abnormal_labevent)
    abnormal_labevent = abnormal_labevent.reset_index(drop=True)
    return abnormal_labevent



###### combine three tables #####
def combine_process(med_pd, diag_pd, pro_pd, labtest_pd):
    """药物、症状、proc的数据结合"""

    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    # 时间
    adm = pd.read_csv(admission_file)
    adm = adm[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']]
    adm.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    adm_key = adm[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    # 病历
    note_pd = pd.read_csv(note_file, usecols=['SUBJECT_ID', 'HADM_ID', 'CATEGORY', 'DESCRIPTION', 'ISERROR', 'TEXT', 'CHARTDATE'])
    note_pd.drop(index = note_pd[note_pd['ISERROR'] == '1'].index, axis=0, inplace=True)
    nt_filtered = note_pd[note_pd['CATEGORY'] == 'Discharge summary']
    nt_filtered.drop(columns=['CATEGORY', 'DESCRIPTION', 'ISERROR'], axis=1, inplace=True)
    note_key = nt_filtered[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    # 五个集合的交集
    combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(adm_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    combined_key = combined_key.merge(note_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    
    diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    adm = adm.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    note = nt_filtered.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    labtest_pd = labtest_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')


    # flatten and merge
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index()  
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['DRUG'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(columns={'ICD9_CODE':'PRO_CODE'})  
    med_pd['DRUG'] = med_pd['DRUG'].map(lambda x: list(x))
    diag_pd['ICD9_CODE'] = diag_pd['ICD9_CODE'].map(lambda x: list(x))
    pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    # data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
    data['DRUG_Len'] = data['DRUG'].map(lambda x: len(x))
    data = adm.merge(data, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    

    # 合并病历
    note = note.groupby(['SUBJECT_ID', 'HADM_ID'])['TEXT'].sum().reset_index()
    note.drop_duplicates(inplace=True)
    note = note.reset_index(drop=True)

    # labtest_pd = labtest_pd.groupby(['SUBJECT_ID', 'HADM_ID'])['test_text'].unique().reset_index()  
    # labtest_pd['test_text'] = labtest_pd['test_text'].map(lambda x: list(x))
    labtest_pd = labtest_pd.groupby(['SUBJECT_ID', 'HADM_ID']).agg({
        'test_text': lambda x: list(x.unique()),
        'test_mnorm': lambda x: list(x.unique()),
        'test_znorm': lambda x: list(x.unique())
    }).reset_index()


    data = data.merge(note, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(labtest_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')


    a = data[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
    a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x:len(x))
    a = a[a['HADM_ID_Len'] > 1]
    data = data.merge(a[['SUBJECT_ID']], on='SUBJECT_ID', how='inner').reset_index(drop=True)
    data = data.sort_values(by=['SUBJECT_ID', 'ADMITTIME'])
    print(data)

    return data




def statistics(data):
    print('#patients ', data['SUBJECT_ID'].unique().shape)
    print('#clinical events ', len(data))
    
    diag = data['ICD9_CODE'].values
    med = data['DRUG'].values
    pro = data['PRO_CODE'].values
    
    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])
    
    print('#diagnosis ', len(unique_diag))
    print('#med ', len(unique_med))
    print('#procedure', len(unique_pro))
    
    avg_diag, avg_med, avg_pro, max_diag, max_med, max_pro, max_lab, cnt, max_visit, avg_visit, avg_labtest = [0 for i in range(11)]

    for subject_id in data['SUBJECT_ID'].unique():
        item_data = data[data['SUBJECT_ID'] == subject_id]
        x, y, z = [], [], []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row['ICD9_CODE']))
            y.extend(list(row['DRUG']))
            z.extend(list(row['PRO_CODE']))
        x, y, z = set(x), set(y), set(z)
        avg_diag += len(x)
        avg_med += len(y)
        avg_pro += len(z)
        avg_labtest += len(list(row['test_text']))
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y) 
        if len(z) > max_pro:
            max_pro = len(z)
        if len(row['test_text']) > max_lab:
            max_lab = len(row['test_text'])
        if visit_cnt > max_visit:
            max_visit = visit_cnt
        
    print('#avg of diagnoses ', avg_diag/ cnt)
    print('#avg of medicines ', avg_med/ cnt)
    print('#avg of procedures ', avg_pro/ cnt)
    print('#avg of labtests ', avg_labtest/ cnt)
    print('#avg of vists ', avg_visit/ len(data['SUBJECT_ID'].unique()))
    
    print('#max of diagnoses ', max_diag)
    print('#max of medicines ', max_med)
    print('#max of procedures ', max_pro)
    print('#max of labtests ', max_lab)
    print('#max of visit ', max_visit)

##### indexing file and final record
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)
                
# create voc set
def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()
    lab_voc = Voc()
    
    for index, row in df.iterrows():
        diag_voc.add_sentence(row['ICD9_CODE'])
        med_voc.add_sentence(row['DRUG'])
        pro_voc.add_sentence(row['PRO_CODE'])
        lab_voc.add_sentence([i.split(':')[0].strip() for i in row['test_text']])
    
    dill.dump(obj={'diag_voc':diag_voc, 'med_voc':med_voc ,'pro_voc':pro_voc, 'lab_voc': lab_voc}, file=open('voc_final.pkl','wb'))
    return diag_voc, med_voc, pro_voc, lab_voc



def create_patient_record(df, diag_voc, med_voc, pro_voc):
    """
    保存list类型的记录
    每一项代表一个患者，患者中有多个visit，每个visit包含三者数组，按顺序分别表示诊断、proc与药物
    存储的均为编号，可以通过voc_final.pkl来查看对应的具体word
    """

    records = [] # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    for subject_id in df['SUBJECT_ID'].unique():
        item_df = df[df['SUBJECT_ID'] == subject_id]
        patient = []
        begin = 0
        for index, row in item_df.iterrows():
            if begin == 0:
                begin = pd.to_datetime(row['ADMITTIME'])
            timestamp = (pd.to_datetime(row['ADMITTIME']) - begin).days # 更改时间戳类型
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row['ICD9_CODE']])
            admission.append([pro_voc.word2idx[i] for i in row['PRO_CODE']])
            admission.append([med_voc.word2idx[i] for i in row['DRUG']])
            admission.append(row['TEXT'])
            admission.append(timestamp)
            patient.append(admission)
        records.append(patient) 
    dill.dump(obj=records, file=open('records_final.pkl', 'wb'))

    return records
        


# get ddi matrix
def get_ddi_matrix(records, diag_voc, proc_voc, med_voc, ddi_file):

    med_voc_size = len(med_voc.idx2word)
    diag_voc_size = len(diag_voc.idx2word)
    proc_voc_size = len(proc_voc.idx2word)
    med_unique_word = [med_voc.idx2word[i] for i in range(med_voc_size)] 



            
    # 加载DDI数据
    ddi_df = pd.read_csv(ddi_file)
    ddi_df['Drug_A'] = ddi_df['Drug_A'].str.lower() # 改为小写
    ddi_df['Drug_B'] = ddi_df['Drug_B'].str.lower() # 改为小写
    ddi_df = ddi_df[(ddi_df['Drug_A'].isin(med_unique_word)) & (ddi_df['Drug_B'].isin(med_unique_word))]

    print(f'ddi lines: {ddi_df.count()}')


    # weighted ehr adj 
    ehr_adj = np.zeros((med_voc_size, med_voc_size))
    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j<=i:
                        continue
                    # ehr_adj[med_i, med_j] = 1
                    # ehr_adj[med_j, med_i] = 1
                    ehr_adj[med_i, med_j] += 1
                    ehr_adj[med_j, med_i] += 1
    dill.dump(ehr_adj, open('ehr_adj_final.pkl', 'wb'))

    dmc_adj = np.zeros((diag_voc_size, med_voc_size))
    for patient in records:
        for adm in patient:
            diag_set = adm[0]
            med_set = adm[2]
            for i, diag_i in enumerate(diag_set):
                for j, med_j in enumerate(med_set):
                    dmc_adj[diag_i, med_j] += 1
    dill.dump(dmc_adj, open('dmc_adj_final.pkl', 'wb'))

    pmc_adj = np.zeros((proc_voc_size, med_voc_size))
    for patient in records:
        for adm in patient:
            proc_set = adm[1]
            med_set = adm[2]
            for i, diag_i in enumerate(proc_set):
                for j, med_j in enumerate(med_set):
                    pmc_adj[proc_set, med_j] += 1
    dill.dump(pmc_adj, open('pmc_adj_final.pkl', 'wb'))

    
    

    ddi_adj = np.zeros((med_voc_size,med_voc_size))
    for index, row in ddi_df.iterrows():
        i = med_voc.word2idx[row['Drug_A']]
        j = med_voc.word2idx[row['Drug_B']]
        if row['Level'] == 'Minor':
            ddi_adj[i, j] = 0.33
            ddi_adj[j, i] = 0.33
        elif row['Level'] == 'Moderate':
            ddi_adj[i, j] = 0.66
            ddi_adj[j, i] = 0.66
        elif row['Level'] == 'Major':
            ddi_adj[i, j] = 1
            ddi_adj[j, i] = 1

    dill.dump(ddi_adj, open('ddi_A_final.pkl', 'wb')) 

    return ddi_adj




# def ddi_rate_score(record, path):
#     # ddi rate
#     if isinstance(path, str):
#         ddi_A = dill.load(open(path, 'rb'))
#     all_cnt = 0
#     dd_cnt = 0
#     for patient in record:
#         for adm in patient:
#             med_code_set = adm[2]
#             for i, med_i in enumerate(med_code_set):
#                 for j, med_j in enumerate(med_code_set):
#                     if j <= i:
#                         continue
#                     all_cnt += 1
#                     if ddi_A[med_i, med_j] != 0 or ddi_A[med_j, med_i] != 0:
#                         dd_cnt += ddi_A[med_i, med_j]
#     if all_cnt == 0:
#         return 0
#     return dd_cnt / all_cnt


def ddi_rate_score(record, path):
    # ddi rate
    if isinstance(path, str):
        ddi_A = dill.load(open(path, 'rb'))
    all_cnt = 0
    dd_cnt = 0
    for patient in record:
        adm = patient[-1]
        #for adm in patient:
        med_code_set = adm[2]
        for i, med_i in enumerate(med_code_set):
            for j, med_j in enumerate(med_code_set):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                    dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt



def icd9_process(data):
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
        diag = row['ICD9_CODE']
        proc = row['PRO_CODE']

        for i in range(len(diag)):
            diag[i] = transICD(diag[i], True)

        for i in range(len(proc)):
            proc[i] = transICD(proc[i], False)

        data.at[index, 'ICD9_CODE'] = diag
        data.at[index, 'PRO_CODE'] = proc
    return data
    


if __name__ == '__main__':
    # MIMIC数据文件，分别包括药物、诊断和proc
    med_file = '../../mimic/mimicd/data/PRESCRIPTIONS.csv'
    diag_file = '../../mimic/mimicd/data/DIAGNOSES_ICD.csv'
    procedure_file = '../../mimic/mimicd/data/PROCEDURES_ICD.csv'
    admission_file = '../../mimic/mimicd/data/ADMISSIONS.csv'
    note_file = '../../mimic/mimicd/data/NOTEEVENTS.csv'
    labevent_file = '../../mimic/mimicd/data/LABEVENTS.csv'
    d_labitem_file = '../../mimic/mimicd/data/D_LABITEMS.csv'

    # 药物信息
    drugbankinfo = "./drugbank_drugs_info.csv"
    NAMEtoSMILES_file = "./NAMEtoSMILES_new.pkl"




    ddi_file = './ddinter.csv'

    # 处理MIMIC中的药物数据
    med_pd = med_process(med_file) # 处方


    druginfo = pd.read_csv(drugbankinfo)
    # NAMEtoSMILES = nametoSMILES(druginfo, med_pd)
    # dill.dump(NAMEtoSMILES, open(NAMEtoSMILES_file, "wb"))
    with open(NAMEtoSMILES_file, 'rb') as Fin:
        NAMEtoSMILES = dill.load(Fin)
    print(f'med_len: {len(med_pd)}')
    med_pd = med_pd[med_pd.DRUG.isin(list(NAMEtoSMILES.keys()))]
    drug_list = med_pd['DRUG'].drop_duplicates().tolist()
    print(f'drug _list len : {len(drug_list)}')
    print(f'med_len: {len(med_pd)}')
    # print('insulin' in drug_list)
    # exit()
    print ('complete medication processing')

    # for diagnosis
    diag_pd = diag_process(diag_file) # 出现次数排名前2000的诊断集合
    print ('complete diagnosis processing')

    # for procedure
    pro_pd = procedure_process(procedure_file)
    print ('complete procedure processing')

    labtest_pd = labtest_process(labevent_file, d_labitem_file, med_pd)
    print ('complete lab test processing')

    # combine
    data = combine_process(med_pd, diag_pd, pro_pd, labtest_pd)
    statistics(data)
    data = icd9_process(data)

    print(data.columns)
    data.to_pickle('data_final.pkl')


    print ('complete combining')
    print(len(data))

    # ddi_matrix
    diag_voc, med_voc, pro_voc, lab_voc = create_str_token_mapping(data)
    records = create_patient_record(data, diag_voc, med_voc, pro_voc)   # diag,proc,medication按顺序存储
    print(len(records))

    data['ADMITTIME'] = pd.to_datetime(data['ADMITTIME'])
    latest_records = data.loc[data.groupby(['SUBJECT_ID'])['ADMITTIME'].idxmax()]

    # 如果需要重置索引，可以添加以下代码
    latest_records = latest_records.reset_index(drop=True)

    print(len(latest_records))
    # 将 latest_records 保存为 CSV 文件
    latest_records.to_pickle('latest_records.pkl')


    ## check
    visit_counts = data.groupby('SUBJECT_ID')['HADM_ID'].nunique()
    print('less than two visit: ', len(visit_counts[visit_counts<2]), len(data))
    # for user in records:
    #     if len(user) <=1:
    #         print('yes!!!',user)

    ddi_adj = get_ddi_matrix(records, diag_voc, pro_voc, med_voc, ddi_file)
    ddi_rate = ddi_rate_score(records, "ddi_A_final.pkl")
    print("ddi_rate", ddi_rate)

