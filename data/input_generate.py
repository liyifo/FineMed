import dill
import pandas as pd


voc_path = "./voc_final.pkl"
with open(voc_path, 'rb') as Fin:
    voc = dill.load(Fin)
diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]

all_data = dill.load(open('./data_final.pkl', "rb"))
print(len(all_data))
print(all_data.columns)


records = [] 
for subject_id in all_data['SUBJECT_ID'].unique():
    # 一个病人 
    item_df = all_data[all_data['SUBJECT_ID'] == subject_id]
    now = []
    
    for index, row in item_df.iterrows():
        if index == 0:
            continue
        patient = [[],[],[],[]]
        for j in range(index+1):
            if j == index:
                input = row['cluster']
                if type(input)!=list:
                    input = eval(input)

                patient[0].append([i[0] for i in input])
                patient[0].append([i[1] for i in input])
                patient[0].append([i[2] for i in input])
                patient[2] = [i[3] for i in input]
                patient[3] = [med_voc.word2idx[i] for i in row['DRUG']]
            else:
                admission = []
                admission.append([diag_voc.word2idx[i] for i in row['ICD9_CODE']])
                admission.append([pro_voc.word2idx[i] for i in row['PRO_CODE']])
                admission.append([med_voc.word2idx[i] for i in row['DRUG']])
        patient[1].append(admission)
        records.append(patient)
# patient = [input, history, label, all_label]
# input = [[diag], [[proc]], [[lab_id], [response]]]
dill.dump(obj=records, file=open('data_final_rein.pkl', 'wb'))