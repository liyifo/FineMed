
import torch
import dill
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
import itertools
from collections import defaultdict
import dgl
import numpy as np
import  torch.nn.functional as F
from sklearn.preprocessing import normalize

def load_mimic3():
    '''
    ddi矩阵，ddi mask矩阵， 
    '''
    data_path = "./data/records_final_rein.pkl"
    voc_path = "./data/voc_final.pkl"

    ddi_adj_path = "./data/ddi_A_final.pkl"
    ddi_mask_path = "./data/ddi_mask_H.pkl"
    ehr_adj_path = './data/ehr_adj_final.pkl'
    dmc_adj_path = './data/dmc_adj_final.pkl'
    pmc_adj_path = './data/pmc_adj_final.pkl'
    #molecule_path = "./data/MIMIC-III/atc3toSMILES.pkl"

    with open(ddi_adj_path, 'rb') as Fin:
        ddi_adj = dill.load(Fin)
    with open(data_path, 'rb') as Fin:
        data = dill.load(Fin)
    # with open(molecule_path, 'rb') as Fin:
    #     molecule = dill.load(Fin)
    with open(voc_path, 'rb') as Fin:
        voc = dill.load(Fin)
    with open(ehr_adj_path, 'rb') as Fin:
        ehr_adj = dill.load(Fin)
    with open(dmc_adj_path, 'rb') as Fin:
        dmc_adj = dill.load(Fin)
    with open(pmc_adj_path, 'rb') as Fin:
        pmc_adj = dill.load(Fin)

    diag_voc, pro_voc, med_voc, lab_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"], voc["lab_voc"]
    voc_size = (
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word),
        len(lab_voc.idx2word)
    )
    
    # 划分数据集
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    data_eval = data[split_point + eval_len :]

    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")
    print(f"Lab num:{len(lab_voc.idx2word)}")
    
    return data_train, data_eval, data_test, voc_size, ddi_adj, ehr_adj, dmc_adj, pmc_adj

