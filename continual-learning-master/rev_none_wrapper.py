# %matplotlib inline
from rev_main_script_wrapper import *
import argparse


parser = argparse.ArgumentParser(description='process input-arguments')
parser.add_argument('--rDataset', type=str, default='sniff-noaug')
parser.add_argument('--rExp', type=int, default=3)
parser.add_argument('--rMeasureType', type=str, default='')
parser.add_argument('--rIsSimpleExp', type=str, default='no')
parser.add_argument('--rEpoch', type=int, default=100)
parser.add_argument('--cuda_num', type=int, default=0)

parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--n_units', type=int, default=32)
parser.add_argument('--lr2', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--trial', type=int, default=1)

parser.add_argument('--exp_setup', type=str, default='')
parser.add_argument('--subject_idx', type=int, default=None)
parser.add_argument('--fisher_n', type=int, default=None)
parser.add_argument('--max_num_cases', type=int, default=5)
parser.add_argument('--cb_test_metric', type=str, default='eval_loss')
parser.add_argument('--bm_on', type=int, default=1)
parser.add_argument('--th', type=float, default=None)



## Load input-arguments
args = parser.parse_args()

print(args)


if args.rExp < 3:
    run_exp12_none(rClassD2List=rev_get_class_D2_list(args),rDataset=args.rDataset,
        rMeasureType=args.rMeasureType,rEpoch=args.rEpoch,cuda_num=args.cuda_num,exp_type=None,
        subject_idx=args.subject_idx,exp_setup=args.exp_setup,
        n_layers=args.n_layers,n_units=args.n_units,lr2=args.lr2,batch_size=args.batch_size,trial=args.trial)
else:
    run_exp3_none_from_t0_permu(rClassD2List=rev_get_class_D2_list(args),rDataset=args.rDataset,
        rMeasureType=args.rMeasureType,rEpoch=args.rEpoch,cuda_num=args.cuda_num,exp_type=None,
        subject_idx=args.subject_idx,exp_setup=args.exp_setup,max_num_cases=args.max_num_cases,
        n_layers=args.n_layers,n_units=args.n_units,lr2=args.lr2,batch_size=args.batch_size,trial=args.trial)

if args.rMeasureType == 'time':
	os.system("rm ../data/saved_model/*measureTime_*")