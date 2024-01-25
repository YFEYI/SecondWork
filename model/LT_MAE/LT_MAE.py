import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.init import xavier_normal_, uniform_, constant_
from model.LT_MAE.layers import TransformerBlock, PositionalEmbedding, CrossAttnTRMBlock, PredictHead,InputProjection,PredictHeadALl


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = 4 * d_model
        layers = args.layers
        dropout = args.dropout
        enable_res_parameter = args.enable_res_parameter
        # TRMs
        self.TRMs = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])

    def forward(self, x):
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
        return x


class Tokenizer(nn.Module):
    def  __init__(self, rep_dim, vocab_size):
        super(Tokenizer, self).__init__()
        # self.center = nn.Linear(rep_dim, vocab_size)
        self.center = nn.Sequential(
            OrderedDict([
                ('linear1', nn.Linear(rep_dim, int(vocab_size/2))),
                ('linear2', nn.Linear(int(vocab_size/2),vocab_size)),
            ])
        )
        self.norm=nn.LayerNorm(normalized_shape=vocab_size)

    def forward(self, x):
        bs, length, dim = x.shape
        x = x.view(-1, dim)
        probs = self.center(x)  # {bs*length, vocab_size}
        ret = F.gumbel_softmax(probs) #{16*259,vocab_size}
        indexes = probs.max(-1, keepdim=True)[1]  # {16*259,128}
        #indexes = ret.max(-1, keepdim=True)[1]  # {16*259,128}
        #{bs,length} 内容是最大值的下标
        return indexes.view(bs, length) # {16,259}


class Regressor(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, layers):
        super(Regressor, self).__init__()
        self.layers = nn.ModuleList(
            [CrossAttnTRMBlock(d_model, attn_heads, d_ffn, enable_res_parameter) for i in range(layers)])

    def forward(self, rep_visible, rep_mask_token):
        for TRM in self.layers:
            rep_mask_token = TRM(rep_visible, rep_mask_token)
        return rep_mask_token


class LT_MAE(nn.Module):
    def __init__(self, args):
        super(LT_MAE, self).__init__()
        self.d_model = args.d_model
        self.linear_proba = True
        self.device = args.device
        self.data_shape = args.data_shape
        self.max_len = self.data_shape[0]
        print(f'{self.max_len}')
        self.mask_len = int(args.mask_ratio * self.max_len)
        self.momentum=args.momentum
        self.position = PositionalEmbedding(args.data_shape[0], args.d_model)
        self.norm = nn.LayerNorm(normalized_shape=self.data_shape[1])

        self.mask_token = nn.Parameter(torch.randn(self.d_model, ),requires_grad=False)
        self.input_projection = InputProjection(l_model=self.data_shape[1], d_model=args.d_model,dropout=args.dropout)
        self.encoder = Encoder(args)
        self.momentum_encoder = Encoder(args)
        self.tokenizer = Tokenizer(self.d_model, args.vocab_size)
        self.reg = Regressor(self.d_model, args.attn_heads, 4 * self.d_model , 1, args.reg_layers)
        # self.predict_head = nn.Linear(d_model, args.num_class)
        self.predict_head = PredictHead(self.d_model,args.data_shape[0],args.num_class, args.dropout)
        self.predict_head_all = PredictHeadALl(args.vocab_size,args.data_shape[0],args.num_class, args.dropout)
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def copy_weight(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_b.data = param_a.data

    def momentum_update(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_b.data = self.momentum * param_b.data + (1 - self.momentum) * param_a.data

    def pretrain_forward(self, x):
        #x = self.norm(x)
        #输出 ：x:Batch, S, d_model}、h_n:{1, d_model}
        x= self.input_projection(x)     #{B,series,d_model}
        x = x.contiguous()
        # 使用x进一步丰富了词向量的长度后，在词向量维度上选取最大值 最后形状{bs,length}
        tokens = self.tokenizer(x)
        # 输入Embedding Z=Z+P
        position = self.position(x)
        x = x + position
        # mask_token是自己创建的可训练参数，大小为{d_model, 1}->{bs, S, d_model} 最后混入位置参数
        #rep_mask_token = self.mask_token.repeat(x.shape[0], x.shape[1], 1) + position

        rep_mask_token = position

        index = np.arange(x.shape[1])
        random.shuffle(index)
        v_index = index[:-self.mask_len]
        m_index = index[-self.mask_len:]  #随机化下标，根据比例划分visible和masked
        visible = x[:, v_index, :]  #Zv
        mask = x[:, m_index, :]   #Zm
        tokens = tokens[:, m_index]  #tokens 一个数字？
        # 混入位置参数后的一组可训练参数,只截取的不可见位置的部分,论文3.4.2中的新初始化向量Zmask
        rep_mask_token = rep_mask_token[:, m_index, :]

        rep_visible = self.encoder(visible)  # 对可见部分使用transformer编码器H  {16, 254*radio, d_model}
        with torch.no_grad():
            rep_mask = self.momentum_encoder(mask)   #对不可见部分使用编码器,(transformer编码器H) encoder==momentum_encoder 本质是一样的

        # 对不可见部分使用解耦cross-attention编码器 F
        # 经过H编码器转换的可见位置Embedding作为输入, 重新初始化的不可见区域的Embedding作为查询
        # 输出rep_mask_prediction: (Sm的? Zm的?) transformer后的上下文表示
        # 解耦模块仅对不可见位置的Embedding进行预测,同时保持可见位置的Embedding未更新。
        rep_mask_prediction = self.reg(rep_visible, rep_mask_token)  #{16, 254*radio, d_model}
        token_prediction_prob = self.tokenizer.center(rep_mask_prediction)  #{16,series,vocab_size}
        return [rep_mask, rep_mask_prediction], [token_prediction_prob, tokens]

    def forward(self, x, total_flags):
        x = self.input_projection(x)  #{16,30,128}
        x = x.contiguous()
        x = x + self.position(x)
        x = self.encoder(x) #{bs,S,d_model}
        if total_flags==True:
            x = self.tokenizer.center(x)  #{B, len, vocab_size}
            x = self.predict_head_all(x)  # {B, num_classes}
            return x  # {bs,S,num_classes}
        #原部分是区分对数据整体分类还是对单独时间步进行分类的判断
        # else:
        #     x = x.reshape(-1, self.d_model)  # {B*len, d_model}
        #     x = self.predict_head(x,perdict = True)   # {B*len, num_classes}
        #现部分是可视化中t-sne，不需要后续分类的分支
        #这样做时间步分类代码会失效，请注意！！！！
        else:
            token = self.tokenizer.center(x)
            return x,token  # {bs,S,num_classes}



    def get_tokens(self, x):
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
        tokens = self.tokenizer(x)
        return tokens
