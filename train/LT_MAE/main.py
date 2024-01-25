import torch
import warnings

warnings.filterwarnings('ignore')
from train.LT_MAE.args import args, Test_data, Train_data_all, Train_data  #执行前需要取消args中相应的注释
from data.dataset import Dataset
from model.LT_MAE.LT_MAE import LT_MAE
from train.LT_MAE.process_all import Trainer
import torch.utils.data as Data
def main():
    # torch.set_num_threads(12)
    torch.cuda.manual_seed(4399)  #44
    train_dataset = Dataset(device=args.device, mode='pretrain', data=Train_data_all)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    args.data_shape = train_dataset.shape()
    train_linear_dataset = Dataset(device=args.device, mode='supervise_train', data=Train_data)
    train_linear_loader = Data.DataLoader(train_linear_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_dataset = Dataset(device=args.device, mode='test', data=Test_data)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)

    print(args.data_shape)
    print('dataset initial ends')
    model = LT_MAE(args)

    print('model initial ends')
    trainer = Trainer(args, model, train_loader, train_linear_loader, test_loader, verbose=True)

    trainer.pretrain()
    trainer.finetune()


if __name__ == '__main__':
    main()
