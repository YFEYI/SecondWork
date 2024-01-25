import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        # 设置一个索引个数为max_len的Embedding查找表，每一个位置长度为d_model
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, enable_res_parameter, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if type(x) == list:
            return self.norm(x[1] + self.dropout(self.a * sublayer(x)))
        if not self.enable:
            return self.norm(x + self.dropout(sublayer(x)))
        else:
            return self.norm(x + self.dropout(self.a * sublayer(x)))


class PointWiseFeedForward(nn.Module):
    """
    FFN implement
    """

    def __init__(self, d_model, d_ffn, dropout=0.1):
        super(PointWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    TRM layer
    """

    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(attn_heads, d_model, dropout)
        self.ffn = PointWiseFeedForward(d_model, d_ffn, dropout)
        self.skipconnect1 = SublayerConnection(d_model, enable_res_parameter, dropout)
        self.skipconnect2 = SublayerConnection(d_model, enable_res_parameter, dropout)

    def forward(self, x, mask):
        x = self.skipconnect1(x, lambda _x: self.attn.forward(_x, _x, _x, mask=mask))
        x = self.skipconnect2(x, self.ffn)
        return x


class CrossAttnTRMBlock(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, dropout=0.1):
        super(CrossAttnTRMBlock, self).__init__()
        self.attn = MultiHeadAttention(attn_heads, d_model, dropout)
        self.ffn = PointWiseFeedForward(d_model, d_ffn, dropout)
        self.skipconnect1 = SublayerConnection(d_model, enable_res_parameter, dropout)
        self.skipconnect2 = SublayerConnection(d_model, enable_res_parameter, dropout)

    def forward(self, rep_visible, rep_mask_token, mask=None):
        x = [rep_visible, rep_mask_token]
        x = self.skipconnect1(x, lambda _x: self.attn.forward(_x[1], _x[0], _x[0], mask=mask))
        x = self.skipconnect2(x, self.ffn)
        return x


class InputProjection(nn.Module):
    def __init__(self,l_model,d_model,dropout):
        super(InputProjection, self).__init__()
        self.Lstm1 = nn.LSTM(input_size=l_model,hidden_size=d_model,num_layers=2,batch_first=True,dropout=dropout)
        self.Lstm2 = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=2, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(normalized_shape=d_model)
        pass
    def forward(self,x):
        x,(_,_) = self.Lstm1(x)  #{B,len,d_model}
        x = self.norm(x)
        return x

class PredictHead(nn.Module):
    def __init__(self, d_model,max_len, num_class, dropout):
        super(PredictHead, self).__init__()
        # self.linear1 = nn.Linear(d_model, d_model // 2)
        # self.linear2 = nn.Linear(d_model//2, d_model)
        # self.linear3 = nn.Linear(max_len * d_model, d_model)
        self.linear_perdict = nn.Linear(d_model,num_class)
        self.drop = nn.Dropout(p=dropout)


        pass
    def forward(self, x, perdict=False):
        #{B*len//2, d_model}
        # x = self.linear1(x) #{B*len//2, d_model//2}
        # x = self.drop(x)
        # x = self.linear2(x) #{B*len//2, d_model}
        # # x = x.flatten(start_dim=1) #{8,15*32}->{8,32}
        # # x = self.linear3(x)
        if perdict == True:
            x = self.linear_perdict(x)  #{B*len//2, d_model}
        return x

class PredictHeadALL(nn.Module):
    def __init__(self, n_cluster,len, num_class, dropout):
        super(PredictHeadALL, self).__init__()
        #self.half_avg_pool1d= nn.AvgPool1d(kernel_size=len,stride=len,padding=0)
        self.linear_perdict = nn.Linear(in_features=len*n_cluster,out_features=num_class)
        pass
    def forward(self,x):
        bs,length,dim = x.shape  #{bs,len,cluster}
        x = x.reshape(bs,-1)  #{bs,len*cluster}
        #x = self.half_avg_pool1d(x)
        x = self.linear_perdict(x)
        return x


class Tokenizer(nn.Module):
    def __init__(self,device):
        super(Tokenizer, self).__init__()
        self.device = device

    def forward(self,x, clusters):
        bs, length, dim = x.shape
        x = x.view(-1, dim)
        x_cluster = clusters.predict(x.cpu().detach().numpy()) # {bs*length, vocab_size}
        cluster_results = torch.from_numpy(x_cluster).float().to(self.device)
        # {bs,length} 内容是最大值的下标
        return cluster_results.view(bs, length)  # {16,259}
