
import pickle
import torch
import numpy as np
from util import *
from metrics import multi_label_metric, ddi_rate_score
from torch.optim import Adam
from collections import defaultdict
import time
import dill
import torch.nn.functional as F
import os
from util import *
import math


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}



def Test(model, model_path, device, data_test, voc_size):
    # with open(model_path, 'rb') as Fin:
    #     model.load_state_dict(torch.load(Fin, map_location=device))
    model = model.to(device).eval()
    print('--------------------Begin Testing--------------------')
    ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
    tic, result, sample_size = time.time(), [], round(len(data_test) * 0.8)
    np.random.seed(0)
    for _ in range(10):
        test_sample = np.random.choice(data_test, sample_size, replace=True)
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = \
            Eval(model, test_sample, voc_size)
        result.append([ddi_rate, ja, avg_f1, prauc, avg_med])
    result = np.array(result)
    mean, std = result.mean(axis=0), result.std(axis=0)
    metric_list = ['ddi_rate', 'ja', 'avg_f1', 'prauc', 'med']
    outstring = ''.join([
        "{}:\t{:.4f} $\\pm$ {:.4f} & \n".format(metric_list[idx], m, s)
        for idx, (m, s) in enumerate(zip(mean, std))
    ])
    print(outstring)
    print('average test time: {}'.format((time.time() - tic) / 10))
    print('parameters', get_n_params(model))



def Eval(model, data_eval, voc_size):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    for step, data in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

        target_output = model(data[0][0],data[0][1], data[0][2], data[1])
        max_logits, max_indices = torch.max(target_output, dim=0)
        max_logits = max_logits.view(1, -1)


        y_gt_tmp = np.zeros(voc_size[2])
        y_gt_tmp[data[3]] = 1
        y_gt.append(y_gt_tmp)

        # prediction prod
        target_output = max_logits.detach().cpu().numpy()[0]
        y_pred_prob.append(target_output)

        # prediction med set
        y_pred_tmp = target_output.copy()
        y_pred_tmp[y_pred_tmp >= 0.5] = 1
        y_pred_tmp[y_pred_tmp < 0.5] = 0
        y_pred.append(y_pred_tmp)

        # prediction label
        y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
        y_pred_label.append(sorted(y_pred_label_tmp))
        visit_cnt += 1
        med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint("\rtest step: {} / {}".format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path="./data/ddi_A_final.pkl")

    llprint(
        "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate*1.0,
            np.mean(ja)*1.0,
            np.mean(prauc)*1.0,
            np.mean(avg_p)*1.0,
            np.mean(avg_r)*1.0,
            np.mean(avg_f1)*1.0,
            med_cnt / visit_cnt,
        )
    )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
    )


    


def Train(model, device, data_train, data_eval, voc_size, args):
    model.to(device=device)
    print('parameters', count_parameters(model))
    optimizer = Adam(list(model.parameters()), lr=args.lr)

    # start iterations
    history = defaultdict(list)
    best_epoch, best_ja = 0, 0
    EPOCH = 200


    for epoch in range(EPOCH):
        tic = time.time()
        print(f'----------------Epoch {epoch}------------------')
        skip_num = 0
        model.train()
        #loss_list = []

        for step, data in enumerate(data_train):

            # data = [input, history, label, all_label], input = [[diag, [proc], [lab_test], [med]], ...]
            # medicine的二维multi-hot编向量，用来计算bce损失
            sub_loss_bce_target = torch.zeros((len(data[0][0]), voc_size[2])).to(device)
            for i in range(len(data[0][0])):
                sub_loss_bce_target[i, data[2][i]] = 1

            all_loss_bce_target = torch.zeros(1, voc_size[2]).to(device)
            all_loss_bce_target[:, data[3]] = 1

            # medicine的二维multi-label向量，用来计算mlm损失
            sub_loss_multi_target = -torch.ones((len(data[0][0]), voc_size[2])).long().to(device)
            for i in range(len(data[0][0])):
                for id, item in enumerate(data[2][i]):
                    sub_loss_multi_target[i, id] = item

            all_loss_multi_target = -torch.ones((1, voc_size[2])).long().to(device)
            for id, item in enumerate(data[3]):
                all_loss_multi_target[0, id] = item

            # 运行模型
            result = model(data[0][0],data[0][1], data[0][2], data[1]) #(d_len, med_num)


            # 子任务loss函数定义
            # print(result, sub_loss_bce_target.shape)
            sub_loss_bce = F.binary_cross_entropy(result, sub_loss_bce_target)
            sub_loss_multi = F.multilabel_margin_loss(result, sub_loss_multi_target)
            

            if args.strategy == 'max':
            # all_task max pooling
                max_logits, max_indices = torch.max(result, dim=0)
                max_logits = max_logits.view(1, -1) # (1, med_num)
                # all_task weighted sum

                all_loss_bce = F.binary_cross_entropy(max_logits, all_loss_bce_target)
                all_loss_multi = F.multilabel_margin_loss(max_logits, all_loss_multi_target)
            elif args.strategy == 'ave':
                ave_logits = torch.sum(result, keepdim=True, dim=0) / result.shape[0]   # (1, med_num)

                all_loss_bce = F.binary_cross_entropy(ave_logits, all_loss_bce_target)
                all_loss_multi = F.multilabel_margin_loss(ave_logits, all_loss_multi_target)

            # multi-hot结果处理
            # result = result.detach().cpu().numpy()[0] # sigmoid
            # result[result >= 0.5] = 1
            # result[result < 0.5] = 0
            # y_label = np.where(result == 1)[0]
            # current_ddi_rate = ddi_rate_score([[y_label]])


            loss = (0.95 * sub_loss_bce + 0.05 * sub_loss_multi) * args.alpha + (1-args.alpha) * (0.95 * all_loss_bce + 0.05 * all_loss_multi)


            #print(f'loss:{epoch}-{loss}')


            # 梯度计算
            optimizer.zero_grad()
            loss.backward() # retain_graph=True
            #print(f'loss:{step}-{idx}-{loss}')
            optimizer.step()

            # del result, loss_ddi, loss
            # torch.cuda.empty_cache()

            llprint("\rtraining step: {} / {}".format(step, len(data_train)))

        print()
        tic2 = time.time()
        print(
            "training time: {}".format(
                tic2 - tic
            )
        )
        #print('loss:', sum(loss_list) / len(loss_list))
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = Eval(
            model, data_eval, voc_size
        )
        print(
            "training time: {}, validate time: {}".format(
                tic2 - tic, time.time() - tic2
            )
        )
        print(f'skip num:{skip_num}')

        history["ja"].append(ja)
        history["ddi_rate"].append(ddi_rate)
        history["avg_p"].append(avg_p)
        history["avg_r"].append(avg_r)
        history["avg_f1"].append(avg_f1)
        history["prauc"].append(prauc)
        history["med"].append(avg_med)

        # 每个epoch的metrics
        if epoch >= 5:
            print(
                "ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}".format(
                    np.mean(history["ddi_rate"][-5:]),
                    np.mean(history["med"][-5:]),
                    np.mean(history["ja"][-5:]),
                    np.mean(history["avg_f1"][-5:]),
                    np.mean(history["prauc"][-5:]),
                )
            )


        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja
            # torch.save(model.state_dict(), os.path.join("saved", "Epoch_{}_JA_{:.4}_DDI_{:.4}.model".format(epoch, ja*1.0, ddi_rate*1.0)))

        print(f"best_epoch: {best_epoch} / {best_ja}")

    dill.dump(history,open(os.path.join("saved", "history.pkl""wb")))