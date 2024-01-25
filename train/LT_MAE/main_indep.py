import json
import os

import torch
import warnings
from data.datautils import load_SEED_indep,load_DEAP_indep
warnings.filterwarnings('ignore')
from train.LT_MAE.args import args
from data.dataset import Dataset
from model.LT_MAE.LT_MAE import LT_MAE
from train.LT_MAE.process_all import Trainer
import torch.utils.data as Data
def main_indep():
    # torch.set_num_threads(12)
    torch.cuda.manual_seed(args.random_seed)  #4399
    _save_path = args.save_path
    for i in range(16):
        args.save_path = _save_path+str(i)+""
        print(f"save_path：{args.save_path}")
        #Train_data_all, Test_data = load_SEED_indep(path=args.data_path, dataset=args.data_set,time_step=args.time_steps, subject=i,seed=44, label_nums=4)
        Train_data_all, Test_data = load_DEAP_indep(path=args.data_path, dataset=args.data_set,time_step=args.time_steps, subject=i,seed=44, label_nums=4)
        train_dataset = Dataset(device=args.device, mode='pretrain', data=Train_data_all)
        train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        args.data_shape = train_dataset.shape()
        train_linear_dataset = Dataset(device=args.device, mode='supervise_train', data=Train_data_all)
        train_linear_loader = Data.DataLoader(train_linear_dataset, batch_size=args.train_batch_size, shuffle=True)
        test_dataset = Dataset(device=args.device, mode='test', data=Test_data)
        test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)
        ####################################### args    ###########################################
        """ 
            原本args里面的保存，由于save_path需要i,所以在这里
        """
        args.eval_per_steps = max(1, int(len(Train_data_all[0]) / args.train_batch_size))  # 你需要读Train_data，你在这里
        args.lr_decay_steps = args.eval_per_steps
        if not os.path.exists(args.save_path):  # 你需要读正确的save_path，你在这里
            os.makedirs(args.save_path)
        config_file = open(args.save_path + '/args.json', 'w')
        tmp = args.__dict__
        json.dump(tmp, config_file, indent=1)
        print(args)
        config_file.close()
        print(args.data_shape)
        print('dataset initial ends')

        model = LT_MAE(args)

        print('model initial ends')
        trainer = Trainer(args, model, train_loader, train_linear_loader, test_loader, verbose=True)

        trainer.pretrain()
        trainer.finetune()


if __name__ == '__main__':
    main_indep()
