import time
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from train.LT_MAE.loss import CE, Align, Reconstruct
from torch.optim.lr_scheduler import LambdaLR


class Trainer():
    def __init__(self, args, model, train_loader, train_linear_loader, test_loader, verbose=False):
        self.args = args
        self.verbose = verbose
        self.device = args.device
        self.print_process(self.device)
        self.model = model.to(torch.device(self.device))
        self.train_loader = train_loader
        self.train_linear_loader = train_linear_loader
        self.test_loader = test_loader
        self.lr_decay = args.lr_decay_rate
        self.lr_decay_steps = args.lr_decay_steps
        self.cr = CE(self.model)
        self.alpha = args.alpha
        self.beta = args.beta
        self.test_cr = torch.nn.CrossEntropyLoss()
        self.num_epoch = args.num_epoch
        self.num_epoch_pretrain = args.num_epoch_pretrain
        self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path
        if self.num_epoch:
            self.result_file = open(self.save_path + '/result.txt', 'w')
            self.result_file.close()
        self.step = 0
        self.best_metric = -1e9
        self.metric = 'acc'

    def pretrain(self):
        print('pretraining')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        eval_acc = 0
        align = Align()
        reconstruct = Reconstruct()
        self.model.copy_weight()
        if self.num_epoch_pretrain:
            print(f'save_path：{self.save_path}')
            result_file = open(self.save_path + '/pretrain_result.txt', 'w')
            result_file.close()
            result_file = open(self.save_path + '/linear_result.txt', 'w')
            result_file.close()
        for epoch in range(self.num_epoch_pretrain):
            self.model.train()
            tqdm_dataloader = tqdm(self.train_loader)
            loss_sum = 0
            loss_mse = 0
            loss_ce = 0
            hits_sum = 0
            NDCG_sum = 0
            for idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                self.optimizer.zero_grad()
                [rep_mask, rep_mask_prediction], [token_prediction_prob, tokens] = self.model.pretrain_forward(batch[0])
                align_loss = align.compute(rep_mask, rep_mask_prediction)
                reconstruct_loss, hits, NDCG = reconstruct.compute(token_prediction_prob, tokens)
                loss_ce = loss_ce + reconstruct_loss.item()
                loss = self.alpha * align_loss + self.beta * reconstruct_loss
                loss_mse = loss_mse + align_loss.item()
                hits_sum =hits_sum+hits.item()
                NDCG_sum =NDCG_sum+NDCG
                loss.backward()
                self.optimizer.step()
                self.model.momentum_update()
                loss_sum = loss_sum + loss.item()
            print('pretrain epoch:{0}, loss:{1}, mse:{2}, ce:{3}, hits:{4}, ndcg:{5}'.format(epoch + 1, loss_sum / (idx + 1),
                                                                                       loss_mse / (idx + 1),
                                                                                       loss_ce / (idx + 1), hits_sum,
                                                                                       NDCG_sum / (idx + 1)))
            result_file = open(self.save_path + '/pretrain_result.txt', 'a+')
            print('pretrain epoch:{0}, loss:{1}, mse:{2}, ce:{3}, hits:{4}, ndcg:{5}'.format(epoch + 1,
                                                                                                  loss_sum / (idx + 1),
                                                                                                  loss_mse / (idx + 1),
                                                                                                  loss_ce / (idx + 1),
                                                                                                  hits_sum,
                                                                                                  NDCG_sum / (idx + 1)),
                  file=result_file)
            result_file.close()
            torch.save(self.model.state_dict(), self.save_path + '/pretrain_model.pkl')

    def finetune(self):
        print('finetune')
        if self.args.load_pretrained_model:
            print('load pretrained model')
            state_dict = torch.load(self.save_path + '/pretrain_model.pkl', map_location=self.device)
            try:
                self.model.load_state_dict(state_dict)
            except:
                model_state_dict = self.model.state_dict()
                for pretrain, random_intial in zip(state_dict, model_state_dict):
                    assert pretrain == random_intial
                    if pretrain in ['input_projection.weight', 'input_projection.bias', 'predict_head.weight',
                                    'predict_head.bias', 'position.pe.weight','tokenizer.center']:
                        state_dict[pretrain] = model_state_dict[pretrain]
                self.model.load_state_dict(state_dict)

        self.model.linear_proba = False
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=self.verbose)
        for epoch in range(self.num_epoch):
            loss_epoch, time_cost = self._train_one_epoch()
            self.result_file = open(self.save_path + '/result.txt', 'a+')
            self.print_process(
                'Finetune epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost))
            print('Finetune train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost),
                  file=self.result_file)
            self.result_file.close()
        self.print_process(self.best_metric)
        return self.best_metric

    def _train_one_epoch(self):
        t0 = time.perf_counter()
        self.model.train()
        tqdm_dataloader = tqdm(self.train_linear_loader) if self.verbose else self.train_linear_loader

        loss_sum = 0
        for idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad()
            loss = self.cr.compute(batch)
            loss_sum += loss.item()
            loss.backward()
            self.optimizer.step()
            self.step += 1
        metric = self.eval_model()
        self.print_process(metric)
        self.result_file = open(self.save_path + '/result.txt', 'a+')
        print('step{0}'.format(self.step), file=self.result_file)
        print(metric, file=self.result_file)
        self.result_file.close()
        if metric[self.metric] >= self.best_metric:
            torch.save(self.model.state_dict(), self.save_path + '/model.pkl')
            self.result_file = open(self.save_path + '/result.txt', 'a+')
            print('saving model of step{0}'.format(self.step), file=self.result_file)
            self.result_file.close()
            self.best_metric = metric[self.metric]
        self.model.train()
        return loss_sum / (idx + 1), time.perf_counter() - t0

    def eval_model(self):
        self.model.eval()
        tqdm_data_loader = tqdm(self.test_loader) if self.verbose else self.test_loader
        metrics = {'acc': 0, 'f1': 0}
        pred_list = []
        label_list = []
        test_loss_sum = 0

        with torch.no_grad():
            for idx, batch in enumerate(tqdm_data_loader):
                batch = [x.to(self.device) for x in batch]
                ret = self.compute_metrics(batch)
                if len(ret) == 2:
                    pred_b, label_b = ret
                    pred_list += pred_b
                    label_list += label_b
                else:
                    pred_b, label_b, test_loss_b = ret
                    pred_b = pred_b.view(-1).cpu().tolist()
                    label_b = label_b.view(-1).cpu().tolist()
                    pred_list.extend(pred_b)
                    label_list.extend(label_b)
                    test_loss_sum += test_loss_b.cpu().item()
            # pred = np.array(pred).flatten()
            # label = np.array(label).flatten()
        confusion_mat = self._confusion_mat(label_list, pred_list)
        self.print_process(confusion_mat)
        self.result_file = open(self.save_path + '/result.txt', 'a+')
        print(confusion_mat, file=self.result_file)
        self.result_file.close()
        if self.args.num_class == 2:
            metrics['f1'] = f1_score(y_true=label_list, y_pred=pred_list)
            metrics['precision'] = precision_score(y_true=label_list, y_pred=pred_list)
            metrics['recall'] = recall_score(y_true=label_list, y_pred=pred_list)
        else:
            metrics['f1'] = f1_score(y_true=label_list, y_pred=pred_list, average='macro')
            metrics['micro_f1'] = f1_score(y_true=label_list, y_pred=pred_list, average='micro')
        metrics['acc'] = accuracy_score(y_true=label_list, y_pred=pred_list)
        metrics['test_loss'] = test_loss_sum / (idx + 1)
        return metrics

    def compute_metrics(self, batch):
        if len(batch) == 2:
            seqs, label = batch
            scores = self.model(seqs, True)
        else:
            seqs1, seqs2, label = batch
            scores = self.model((seqs1, seqs2), True)

        one_hot_label = F.one_hot(label.to(torch.int64), num_classes=4).float()
        one_hot_label = one_hot_label.unsqueeze(dim=1).repeat(1, scores.shape[0] // seqs.shape[0], 1).reshape(
            -1, 4)
        test_loss = self.test_cr(scores, one_hot_label)
        _, pred = torch.topk(scores, 1)  # {16,1}
        _, label = torch.topk(one_hot_label, 1)  # {16,1}
        #pred = pred.view(-1)
        return pred.squeeze(), label.squeeze(), test_loss

    def _confusion_mat(self, label, pred):
        mat = np.zeros((self.args.num_class, self.args.num_class))
        for _label, _pred in zip(label, pred):
                mat[_label, _pred] += 1
        return mat

    def print_process(self, *x):
        if self.verbose:
            print(*x)
