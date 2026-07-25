
import torch
import torch.nn as nn
from typing import List
import dgl
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
import dgl.function as fn


class GraphConvolution(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphAttentionV2Layer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False):
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(out_features, self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, h: torch.Tensor, nei_embedding: torch.Tensor, adj_mat: torch.Tensor, use_elu:bool):
        '''
        h: (nodes, embedding)
        nei_embedding: (nei_nodes, embedding)
        adj_mat: (nodes, nei_nodes)
        '''
        adj_mat = adj_mat.T.unsqueeze(2) # 1 or equal to n_heads

        n_nodes = h.shape[0]
        nei_nodes = nei_embedding.shape[0]
        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(nei_embedding).view(nei_nodes, self.n_heads, self.n_hidden)
        g_l_repeat = g_l.repeat(nei_nodes, 1, 1) # (n_nodes*nei_nodes, self.n_heads, self.n_hidden)
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0) # (nei_nodes*n_nodes, self.n_heads, self.n_hidden)
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(n_nodes, nei_nodes, self.n_heads, self.n_hidden)
        e = self.attn(self.activation(g_sum)) # (n_nodes, nei_nodes, self.n_heads, 1)
        e = e.squeeze(-1) # (n_nodes, nei_nodes, self.n_heads)
        # print(n_nodes, nei_nodes, adj_mat.shape)
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == nei_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        e = e.masked_fill(adj_mat == 0, float('-inf'))
        # changed softmax
        exp_attention = torch.exp(e)
        weighted_exp_attention = exp_attention * adj_mat
        softmax_denominator = torch.sum(weighted_exp_attention, dim=1, keepdim=True) # (n_nodes, 1, self.n_heads)
        a = weighted_exp_attention / softmax_denominator
        # a = self.softmax(e)

        a = self.dropout(a)
        attn_res = torch.einsum('ijh,jhf->ihf', a, g_r) #  (n_nodes, nei_nodes, self.n_heads)  (nei_nodes, self.n_heads, self.n_hidden) -> (n_nodes, self.n_heads, self.n_hidden)
        if use_elu:
            attn_res = torch.nn.functional.elu(attn_res)
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=1)


class GCN(torch.nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        ehr_adj = self.normalize(ehr_adj + np.eye(ehr_adj.shape[0]))
        ddi_adj = self.normalize(ddi_adj + np.eye(ddi_adj.shape[0]))

        self.ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
        self.gcn3 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)

        ehr_node_embedding = F.relu(ehr_node_embedding)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)

        ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ehr_node_embedding, ddi_node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class GATv2(torch.nn.Module):
    def __init__(self, voc_size, emb_dim, dmc_adj, pmc_adj, device=torch.device('cpu:0')):
        super(GATv2, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device


        self.ehr_adj = torch.FloatTensor(dmc_adj).to(device)
        self.ddi_adj = torch.FloatTensor(pmc_adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn11 = GraphAttentionV2Layer(voc_size, emb_dim, 4)
        self.gcn12 = GraphAttentionV2Layer(voc_size, emb_dim, 4)
        self.gcn2 = GraphAttentionV2Layer(emb_dim, emb_dim, 1)
        self.gcn3 = GraphAttentionV2Layer(emb_dim, emb_dim, 1)

    def forward(self, diag_embedding, proc_embedding):
        ehr_node_embedding = self.gcn11(self.x, diag_embedding, self.ehr_adj, True)
        ddi_node_embedding = self.gcn12(self.x, proc_embedding, self.ddi_adj, True)

        ehr_node_embedding = self.gcn2(ehr_node_embedding, diag_embedding, self.ehr_adj, False)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, proc_embedding, self.ddi_adj, False)
        return ehr_node_embedding, ddi_node_embedding




class SelfAttend(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(SelfAttend, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(embedding_size, 32),
            nn.Tanh()
        )
        
        self.gate_layer = nn.Linear(32, 1)

    def forward(self, seqs):
        """
        :param seqs: shape [batch_size, seq_length, embedding_size]
        :param seq_lens: shape [batch_size, seq_length]
        :return: shape [batch_size, seq_length, embedding_size]
        """
        gates = self.gate_layer(self.h1(seqs)).squeeze(-1) # （len）
        p_attn = F.softmax(gates, dim=-1)
        p_attn = p_attn.unsqueeze(-1) #（len, 1）
        h = seqs * p_attn # (len, dim)
        output = torch.sum(h, dim=0) # 
        return output


class RSMed(nn.Module):
    def __init__(self, voc_size: list, ehr_adj, ddi_adj, dmc_adj, pmc_adj,
                 embedding_dimension: int,
                 embedding_dropout: float,
                 device:int):
        """
        [[diag, [proc], [lab]], [] , []]
        :param num_items: int, number of items
        :param num_users: int, number of users
        :param hop_num: int, , number of hops
        :param embedding_dimension: int, dimension of embedding
        :param temporal_feature_dimension: int, the input dimension of temporal feature
        :param embedding_dropout: float, embedding dropout rate
        :param temporal_attention_dropout: float, temporal attention dropout rate
        :param temporal_information_importance: float, importance of temporal information
        """
        super(RSMed, self).__init__()

        self.voc_size = voc_size
        self.embedding_dimension = embedding_dimension

        self.diagnosis_embedding = nn.Sequential(
            nn.Embedding(voc_size[0], embedding_dimension),
            nn.Dropout(embedding_dropout)
        )
        self.procedure_embedding = nn.Sequential(
            nn.Embedding(voc_size[1], embedding_dimension),
            nn.Dropout(embedding_dropout)
        )

        self.lab_embedding = nn.Sequential(
            nn.Embedding(voc_size[3], embedding_dimension),
            nn.Dropout(embedding_dropout)
        )
        self.med_level_linear = nn.Linear(embedding_dimension, embedding_dimension)

        # self.dmc_input = self.convert_to_gat_adj_matrix(dmc_adj)
        # self.pmc_input = self.convert_to_gat_adj_matrix(pmc_adj)
        # if GAT == False:
        self.med_gcn =  GCN(voc_size=voc_size[2], emb_dim=embedding_dimension, ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device)
        self.med_gat = GATv2(voc_size=voc_size[2], emb_dim=embedding_dimension, dmc_adj=dmc_adj, pmc_adj=pmc_adj, device=device)
        # else:
        #     pass
        # self.med_gcn =  GATv2(voc_size=voc_size[2], emb_dim=embedding_dimension, ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device)
        self.p_attn_w  = nn.Linear(embedding_dimension, embedding_dimension)
        self.l_attn_w  = nn.Linear(embedding_dimension, embedding_dimension)

        self.diag_self_attend = SelfAttend(embedding_dimension)
        self.proc_self_attend = SelfAttend(embedding_dimension)
        self.comorb_linear = nn.Linear(3*embedding_dimension, embedding_dimension)
        self.patient_linear = nn.Linear(4*embedding_dimension, embedding_dimension)
        self.med_linear = torch.nn.Sequential(
            # torch.nn.ReLU(), # can remove
            nn.Linear(4*embedding_dimension, embedding_dimension)
        ) 

        self.scale = embedding_dimension ** 0.5

        # self.leaky_relu_func = nn.LeakyReLU(negative_slope=0.2)


        self.final_fcn = torch.nn.Sequential(
            # torch.nn.ReLU(), # can remove
            torch.nn.Linear(embedding_dimension, voc_size[2])
        )

        self.history_linear = nn.Linear(embedding_dimension,1) 

        self.inter = torch.nn.Parameter(torch.FloatTensor(1))
        self.inter2 = torch.nn.Parameter(torch.FloatTensor(1))
        self.device = device
        self.init_weights()

    def convert_to_gat_adj_matrix(self, cooccurrence_matrix):
        diag_num, med_num = cooccurrence_matrix.shape
        
        # 初始化一个 (diag_num + med_num, diag_num + med_num) 的零矩阵
        adj_matrix = np.zeros((diag_num + med_num, diag_num + med_num))
        
        # 将共现矩阵放在邻接矩阵的右上部分（诊断到药物）
        adj_matrix[:diag_num, diag_num:] = cooccurrence_matrix
        
        # 将共现矩阵的转置放在邻接矩阵的左下部分（药物到诊断）
        adj_matrix[diag_num:, :diag_num] = cooccurrence_matrix.T
        
        return adj_matrix


    def forward(self, diag, proc, labtest, history):
        '''
        diag: (*)
        proc: (diag_num, *)
        labtest: (diag_num, *)
        

        now_visit:[[diag,[proc],[labtest]],[],[]]
        history: [[diag, proc, medication, labtest], [], []]

        '''
        diag_tensor = torch.LongTensor(diag).to(self.device)
        diag_base_embeddings = self.diagnosis_embedding(diag_tensor) # (d_len, dim)

        procedures_embeddings, labtests_embeddings, severity_reprs = [], [], []
        for index, diag_embedding in enumerate(diag_base_embeddings):
            current_proc = proc[index]
            if len(current_proc) == 0:
                proc_repr = torch.zeros(self.embedding_dimension, device=self.device)
            else:
                proc_tensor = torch.LongTensor(current_proc).to(self.device)
                proc_embed = self.procedure_embedding(proc_tensor) # (p_len, dim)
                proc_attn_scores = torch.matmul(proc_embed, self.p_attn_w(diag_embedding)) / self.scale
                proc_attn = torch.softmax(proc_attn_scores, dim=0)
                proc_repr = torch.sum(proc_attn.unsqueeze(-1) * proc_embed, dim=0)
            procedures_embeddings.append(proc_repr)

            current_lab_ids, current_lab_vals = labtest[index]
            if len(current_lab_ids) == 0:
                lab_repr = torch.zeros(self.embedding_dimension, device=self.device)
            else:
                lab_ids_tensor = torch.LongTensor(current_lab_ids).to(self.device)
                lab_values = torch.as_tensor(current_lab_vals, dtype=torch.float, device=self.device)
                lab_embed = self.lab_embedding(lab_ids_tensor) # (l_len, dim)
                lab_embed = lab_embed * lab_values.unsqueeze(-1)
                lab_attn_scores = torch.matmul(lab_embed, self.l_attn_w(diag_embedding)) / self.scale
                lab_attn = torch.softmax(lab_attn_scores, dim=0)
                lab_repr = torch.sum(lab_attn.unsqueeze(-1) * lab_embed, dim=0)
            labtests_embeddings.append(lab_repr)

            severity_reprs.append(torch.cat([diag_embedding, proc_repr, lab_repr], dim=-1)) # (3*dim)

        procedures_embeddings = torch.stack(procedures_embeddings, dim=0) # (d_len, dim)
        labtests_embeddings = torch.stack(labtests_embeddings, dim=0) # (d_len, dim)
        severity_reprs = torch.stack(severity_reprs, dim=0) # (d_len, 3*dim)

        if severity_reprs.shape[0] > 1:
            comorb_raw = (severity_reprs.sum(dim=0, keepdim=True) - severity_reprs) / (severity_reprs.shape[0] - 1)
        else:
            comorb_raw = torch.zeros_like(severity_reprs)
        comorb_embeddings = self.comorb_linear(comorb_raw) # (d_len, dim)
        diagnoses_embeddings = self.patient_linear(torch.cat([severity_reprs, comorb_embeddings], dim=1)) # (d_len, dim)

        # 药物表征
        ehr_embedding, ddi_embedding = self.med_gcn() 
        # ehr_embedding, ddi_embedding = self.med_gcn(self.medication_embedding(torch.LongTensor([i for i in range(self.voc_size[2])]).to(self.device))) 
        # drug_embedding = self.medication_embedding(torch.LongTensor([i for i in range(self.voc_size[2])]).to(self.device)) + ehr_embedding - ddi_embedding * self.inter # (med_num, embedding_dimension)
        # drug_embedding = ehr_embedding - ddi_embedding * self.inter # (med_num, embedding_dimension)
        dmc_embedding, pmc_embedding = self.med_gat(self.diagnosis_embedding(torch.LongTensor([i for i in range(self.voc_size[0])]).to(self.device)), self.procedure_embedding(torch.LongTensor([i for i in range(self.voc_size[1])]).to(self.device)))
        drug_embedding = self.med_linear(torch.cat([ehr_embedding-self.inter2*ddi_embedding, dmc_embedding, pmc_embedding], dim=1))

        # 拼接（编码表征，程序表征，实验室测试信息，历史信息）
       


        # 历史信息 1.直接相加编码  2.自注意力编码
        history_diag = [i[0] for i in history] # (visit_num, *)
        history_proc = [i[1] for i in history]
        history_diag = torch.cat([torch.sum(self.diagnosis_embedding(torch.LongTensor([i]).to(self.device)), keepdim=True, dim=1) for i in history_diag], dim=0) # (visit_num, emb_dim)
        history_proc = torch.cat([torch.sum(self.procedure_embedding(torch.LongTensor([i]).to(self.device)), keepdim=True, dim=1) for i in history_proc], dim=0)
        history_diag = self.diag_self_attend(history_diag) # (visit_len, emb)
        history_proc = self.proc_self_attend(history_proc)
        
        # 根据历史信息计算药物的推荐概率
        visit_level = torch.softmax(diagnoses_embeddings@history_diag.transpose(-2,-1) + procedures_embeddings@history_proc.transpose(-2,-1)/self.embedding_dimension**0.5, dim=-1) # (d_len,visit_num)
        med_level = torch.softmax(self.med_level_linear(diagnoses_embeddings)@drug_embedding.transpose(-2,-1) /self.embedding_dimension**0.5, dim=-1) # (d_len, med_num)
        total_level = visit_level.unsqueeze(2) @ med_level.unsqueeze(1) # (d_len, visit_num, 1) * (d_len, 1, med_num) = (d_len, visit_num, med_num)
        total_level = total_level.sum(dim=1) # (d_len, med_num)
        row_scores = total_level.sum(dim=-1, keepdim=True) # (d_len, 1)
        total_level = total_level / row_scores # (d_len, med_num)
        # # print(total_level)


        # Recommendation
        logit = F.sigmoid(self.final_fcn(F.softmax(diagnoses_embeddings@drug_embedding.transpose(0,1), dim=-1)@drug_embedding + diagnoses_embeddings)) #(d_len, med_num)

        # 结合历史信息
        weight = F.sigmoid(self.history_linear(diagnoses_embeddings))
        logit = weight*logit + (1-weight)*total_level
        # print(weight)
        # write a function to calculate the logit
        # logit = torch.sigmoid(logit) # (d_len, med_num)

        return logit




    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1

        self.diagnosis_embedding[0].weight.data.uniform_(-initrange, initrange)
        self.procedure_embedding[0].weight.data.uniform_(-initrange, initrange)
        self.lab_embedding[0].weight.data.uniform_(-initrange, initrange)

