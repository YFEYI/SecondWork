import argparse
import os

import json
from data.datautils import load_SEED,load_DEAP
from data.datautils import load_SEED,load_SEED_indep

parser = argparse.ArgumentParser()
################################## SEED ##########################################0
# parser.add_argument('--data_set', type=str, default='seed')
# parser.add_argument('--data_path', type=str,
#                     default='../../data/seed/')
# parser.add_argument('--train_ratio', type=float, default=0.8)
# parser.add_argument('--fine_ratio', type=float, default=0.8)
# parser.add_argument('--test_ratio', type=float, default=0.2)
# parser.add_argument('--time_steps', type=float, default=1)  #(重要参数) 0.5、1 、2、 3、4
# parser.add_argument('--save_path', type=str, default='./visualization/SEED/')       #step_mask
# #parser.add_argument('--save_path', type=str, default='./visualization/SEED/independent/')       #independent
# #parser.add_argument('--save_path', type=str, default='./visualization/SEED/dependent/')        #dependent

################################## DEAP ##########################################
parser.add_argument('--data_set', type=str, default='deap')
parser.add_argument('--data_path', type=str,
                    default='../../data/deap/')
parser.add_argument('--train_ratio', type=float, default=0.8)
parser.add_argument('--fine_ratio', type=float, default=0.2)
parser.add_argument('--test_ratio', type=float, default=0.2)
parser.add_argument('--time_steps', type=float, default=0.5)      #(重要参数) 0.5、1 、2、 3、4
parser.add_argument('--save_path', type=str, default='./visualization/DEAP/step_mask')       #step_mask
#parser.add_argument('--save_path', type=str, default='./visualization/DEAP/independent/')       #independent
#parser.add_argument('--save_path', type=str, default='./visualization/DEAP/dependent/')        #dependent

################################## 通用参数 ##########################################

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--train_batch_size', type=int, default=16)
parser.add_argument('--test_batch_size', type=int, default=16)

parser.add_argument('--random_seed', type=int, default=4399)  #44
parser.add_argument('--num_class', type=int, default=4)

# model args
parser.add_argument('--d_model', type=int, default=128)  #128
#parser.add_argument('--d_series', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.2) #0.5
parser.add_argument('--attn_heads', type=int, default=4)  #4
parser.add_argument('--eval_per_steps', type=int, default=16)
parser.add_argument('--enable_res_parameter', type=int, default=1)
parser.add_argument('--layers', type=int, default=8)  #8
parser.add_argument('--alpha', type=float, default=5)  #5
parser.add_argument('--beta', type=float, default=1)   #1

parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--vocab_size', type=int, default=64)  #196v #64
parser.add_argument('--mask_ratio', type=float, default=0.4)  #0.6（重要参数）0.2、0.4、0.6、0.8
parser.add_argument('--reg_layers', type=int, default=2)

# train args
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_decay_rate', type=float, default=1.)
parser.add_argument('--lr_decay_steps', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--num_epoch', type=int, default=50)  #150
#pretrain args
parser.add_argument('--num_epoch_pretrain', type=int, default=50)
parser.add_argument('--load_pretrained_model', type=int, default=1)

args = parser.parse_args()

################################## DEAP ##########################################
Train_data_all, Train_data, Test_data= load_DEAP(
    path=args.data_path, dataset = args.data_set, time_step = args.time_steps,
    train_ratio=args.train_ratio, fine_ratio=args.fine_ratio, test_ratio=args.test_ratio, seed=args.random_seed, label_num = 4)
args.eval_per_steps = max(1, int(len(Train_data[0]) / args.train_batch_size))  #你需要读Train_data，你在这里
args.lr_decay_steps = args.eval_per_steps
if not os.path.exists(args.save_path):  #你需要读正确的save_path，你在这里
    os.makedirs(args.save_path)
config_file = open(args.save_path + '/args.json', 'w')
tmp = args.__dict__
json.dump(tmp, config_file, indent=1)
print(args)
config_file.close()
################################## DEAP_INDEPENDENT ##########################################0
"""
    需要循环这一部分的dep和indep任务，将循环写在了main方法里，这样就需要注释上面的数据读取
"""
################################## SEED ##########################################0
# Train_data_all, Train_data, Test_data= load_SEED(
#     path=args.data_path, dataset = args.data_set, time_step = args.time_steps,
#     train_ratio=args.train_ratio, fine_ratio=args.fine_ratio, test_ratio=args.test_ratio,seed=args.random_seed, label_nums = 4)
# args.eval_per_steps = max(1, int(len(Train_data[0]) / args.train_batch_size))  #你需要读Train_data，你在这里
# args.lr_decay_steps = args.eval_per_steps
# if not os.path.exists(args.save_path):  #你需要读正确的save_path，你在这里
#     os.makedirs(args.save_path)
# config_file = open(args.save_path + '/args.json', 'w')
# tmp = args.__dict__
# json.dump(tmp, config_file, indent=1)
# print(args)
# config_file.close()
################################## SEED_INDEPENDENT ##########################################0
"""
    需要循环这一部分的dep和indep任务，将循环写在了main方法里，这样就需要注释上面的数据读取
"""

