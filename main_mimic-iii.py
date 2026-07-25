
import torch
import argparse
from train import *
from model import *
from load_data import load_mimic3

# Parameters setting
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", default=False, help="test mode")
parser.add_argument("--resume_path", type=str, default='./saved/Epoch_34_JA_0.4278.model', help="resume path")
parser.add_argument("--device", type=int, default=1, help="gpu id to run on, negative for cpu")
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument('--dp', default=0.2, type=float, help='dropout ratio')
parser.add_argument('--emb_dim', type=int, default=64, help='embedding dimension size')
parser.add_argument('--alpha', type=float, default=0.5, help='subtask importance')
parser.add_argument('--strategy', type=str, default='ave', help='training strategy')




args = parser.parse_args()
    


torch.manual_seed(1203)
np.random.seed(2048)

def get_model_name(args):
    model_name = [
        f'dim_{args.emb_dim}',  f'lr_{args.lr}', f'coef_{args.kp}',
        f'dp_{args.dp}', f'ddi_{args.target_ddi}'
    ]
    if args.embedding:
        model_name.append('embedding')
    return '-'.join(model_name)

# run framework
def main():
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    data_train, data_eval, data_test, voc_size, ddi_adj, ehr_adj, dmc_adj, pmc_adj = load_mimic3()

    model = JustMed(voc_size, ddi_adj, ehr_adj, dmc_adj, pmc_adj, args.emb_dim, args.dp, device).to(device)

    print('voc_size:', voc_size)
    if args.test:
        Test(model, args.resume_path, device, data_test, voc_size)
    else:
        Train(model, device, data_train, data_eval, voc_size, args)




if __name__ == '__main__':
    main()

