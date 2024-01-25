import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.utils.data as Data
from model.LT_MAE.LT_MAE import LT_MAE
from train.LT_MAE.args import args, Train_data_all
from dataset import Dataset
from sklearn.manifold import TSNE

def load_model():
    model = LTimeMAE(args).cuda()
    state_dict = torch.load('./visualization/try/model.pkl', map_location=args.device)
    model.load_state_dict(state_dict)
    model.linear_proba = False
    return model
def load_data():
    # 绘制分为两部分
    # 1、使用特征提取过后的数据和扩展后的标签
    # 2、使用表征学习后的数据和扩展后的标签
    # 2、使用表征学习后的数据tokenizer之后的标签
    test_dataset = Dataset(device=args.device, mode='test', data=Train_data_all)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)
    args.data_shape = test_dataset.shape()
    return test_loader

def tsne_drawer_pure(data_loader):
    model=load_model()
    model.eval()
    reps = []
    labels = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):
            seqs, label = batch     ##原始数据 {16,259,124}  {16, }
            label = label.unsqueeze(1)  #{16, }->{16,1}
            label = label.repeat(1, seqs.shape[1], 1)
            label = label.reshape(-1, 1)#{16,1}->{16,259,1}->{4144,1}
            seqs = seqs.reshape(-1,seqs.shape[2]) #{4144,124}
            for i in range(len(seqs)):
                reps.append(seqs[i].cpu().numpy())
                labels.append(label[i].cpu().numpy())
        reps = np.array(reps)
        labels = np.array(labels)
    # tsne = TSNE(n_components=2, perplexity=16, learning_rate=200, random_state=44)#SEED
    tsne = TSNE(n_components=2, perplexity=4, learning_rate=100, random_state=44)#DEAP
    rep_new = tsne.fit_transform(reps)
    np.save('pic/epilepsy/tsne_pure_data.npy', rep_new)
    x = rep_new[:, 0]
    y = rep_new[:, 1]
    labels = labels.squeeze()
    plt.scatter(rep_new[:, 0], rep_new[:, 1], c=labels, s=5)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(ls='--')
    plt.savefig('./pic/epilepsy/tsne_pure_data.svg', format='svg')
    plt.show()

def tsne_drawer_tokenizer(data_loader):
    model=load_model()
    model.eval()
    data_loader = load_data()
    reps = []
    labels = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):
            seqs, label = batch
            rep_batch,token = model(seqs,False) #{16,259,128}
            token = token.reshape(-1, token.shape[2])#{16,259,64}->{4144,64}
            rep_batch = rep_batch.reshape(-1, rep_batch.shape[2])#{4144,128}
            token = token.argmax(axis=-1)
            for i in range(len(rep_batch)):
                reps.append(rep_batch[i].cpu().numpy())
                labels.append(token[i].cpu().numpy())
        reps = np.array(reps)
        labels = np.array(labels)
    #tsne = TSNE(n_components=2, perplexity=16, learning_rate=200, random_state=44) #SEED
    tsne = TSNE(n_components=2, perplexity=4, learning_rate=150, random_state=44)  #DEAP
    rep_new = tsne.fit_transform(reps)
    np.save('pic/epilepsy/tsne_tokenizer_data.npy', rep_new)
    labels = labels.squeeze()

    plt.scatter(rep_new[:, 0], rep_new[:, 1], c=labels, s=5)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(ls='--')
    plt.savefig('./pic/epilepsy/tsne_tokenizer_data.svg', format='svg')
    plt.show()

def tsne_drawer_encoder(data_loader):
    model=load_model()
    model.eval()
    data_loader = load_data()
    reps = []
    labels = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):
            seqs, label = batch
            rep_batch,token = model(seqs,False) #{16,259,128}
            label = label.unsqueeze(1)  # {16, }->{16,1}
            label = label.repeat(1, seqs.shape[1], 1)
            label = label.reshape(-1, 1)  # {16,1}->{16,259,1}->{4144,1}
            rep_batch = rep_batch.reshape(-1, rep_batch.shape[2])#{4144,128}
            for i in range(len(rep_batch)):
                reps.append(rep_batch[i].cpu().numpy())
                labels.append(label[i].cpu().numpy())
        reps = np.array(reps)
        labels = np.array(labels)
    #tsne = TSNE(n_components=2, perplexity=16, learning_rate=200, random_state=44)  #SEED
    tsne = TSNE(n_components=2, perplexity=4, learning_rate=150, random_state=44)   #DEAP
    rep_new = tsne.fit_transform(reps)
    np.save('pic/epilepsy/tsne_encoder_data.npy', rep_new)
    labels = labels.squeeze()

    plt.scatter(rep_new[:, 0], rep_new[:, 1], c=labels, s=5)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(ls='--')
    plt.savefig('./pic/epilepsy/tsne_encoder_data.svg', format='svg')
    plt.show()

# for idx, batch in enumerate(tqdm(test_loader)):
#     seqs, label = batch
#     # label = label.cpu().numpy()
#     # rep_batch = model(seqs)
#     for i in range(len(seqs)):
#         reps.append(seqs[i].cpu().numpy())
#         labels.append(label[i])
# reps = np.array(reps)
if __name__ == '__main__':
    data_loader = load_data()
    #tsne_drawer_pure(data_loader)
    tsne_drawer_tokenizer(data_loader)
    tsne_drawer_encoder(data_loader)


