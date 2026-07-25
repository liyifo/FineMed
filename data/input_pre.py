import dill
import pandas as pd



voc_path = "./voc_final.pkl"
with open(voc_path, 'rb') as Fin:
    voc = dill.load(Fin)
diag_voc, pro_voc, med_voc, lab_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"], voc["lab_voc"]

last_data = dill.load(open('./data_final.pkl', "rb"))
print(len(last_data))
print(last_data.columns)


for index, row in last_data.iterrows():
    if type(row['cluster']) !=list:
        cluster = eval(row['cluster'])
    else:
        cluster = row['cluster']

    lab_dict = {}
    for i in row['test_mnorm']:
        key, value = i.split(':')
        value = float(value)
        lab_dict[key] = value

    new_cluster = []
    for i in range(len(cluster)):
        diag = [0,[], [[],[]], []]
        #print(i[0], diag_voc.word2idx[i[0]])
        diag_name = cluster[i][0]
        if type(diag_name)==str:
            diag_name = diag_name.strip()
        else:
            diag_name = str(diag_name)
        if diag_name[0] == '[' and diag_name[-1] == ']':
            diag_name = diag_name[2:-2]
  
        if diag_name=='8.45':
            diag_name='008.45'
        elif diag_name=='288.6':
            diag_name='288.60'
        elif diag_name=='250.0':
            diag_name='250.00'
        elif diag_name=='574.2':
            diag_name='574.20'
        elif diag_name=='70.54':
            diag_name='070.54'
        elif diag_name=='54.1':
            diag_name='054.10'
        elif diag_name=='41.85':
            diag_name='041.85'
        elif diag_name=='250.6':
            diag_name='250.60'
        elif diag_name=='428.3':
            diag_name='428.30'
        elif diag_name=='562.1':
            diag_name='562.10'
        elif diag_name=='288.5':
            diag_name='288.50'

        diag[0] = diag_voc.word2idx[diag_name]


        for j in cluster[i][1]:
            if j not in pro_voc.word2idx.keys():
                continue
            diag[1].append(pro_voc.word2idx[j])

        key = [j.split(':')[0].strip() for j in cluster[i][2]]

        for j in key:
            if j not in lab_dict.keys():
                continue
            diag[2][0].append(lab_voc.word2idx[j])
            diag[2][1].append(lab_dict[j])
        new_cluster.append(diag)
    last_data.at[index, 'cluster'] = new_cluster

    if type(row['sub_label']) !=list:
        sub_label = eval(row['sub_label'])
    else:
        sub_label = row['sub_label']
    new_label = [i[0] for i in new_cluster]
    for label in sub_label:
        # [med ,[diag], [proc]]
        # 先转化为icd9编码
        diag = [0,[], []]
        med_name = label[0]
        if type(med_name)==str:
            med_name = med_name.strip()
        med_id = med_voc.word2idx[med_name]

        for i in label[1]:
            if i in diag_voc.word2idx.keys():
                for diag in new_cluster:
                    if diag[0] == diag_voc.word2idx[i]:
                        diag[3].append(med_id)
            elif i in pro_voc.word2idx.keys():
                for diag in new_cluster:
                    for proc in diag[1]:
                        if proc == pro_voc.word2idx[i]:
                            diag[3].append(med_id)
                            
        #try:
        if len(label)==3:
            for i in label[2]:
                if i in pro_voc.word2idx.keys():
                    for diag in new_cluster:
                        for proc in diag[1]:
                            if proc == pro_voc.word2idx[i]:
                                diag[3].append(med_id)
                                #print('yes')
                elif i in diag_voc.word2idx.keys():
                    for diag in new_cluster:
                        if diag[0] == diag_voc.word2idx[i]:
                            diag[3].append(med_id)
        
        # except Exception as e:
        #     print(e)
    for diag in new_cluster:
        diag[3] = list(set(diag[3]))
    last_data.at[index, 'cluster'] = new_cluster


dill.dump(obj=last_data, file=open('data_final.pkl', 'wb'))
