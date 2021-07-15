# -*- encoding: utf-8 -*-
'''
@File    :   crf.py
@Time    :   2019/11/23 17:35:36
@Author  :   Cao Shuai
@Version :   1.0
@Contact :   caoshuai@stu.scu.edu.cn
@License :   (C)Copyright 2018-2019, MILAB_SCU
@Desc    :   None
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from pytorch_pretrained_bert import BertModel
import math
from transformers import AutoModel, AlbertModel


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp_batch(log_Tensor, axis=-1):  # shape (batch_size,n,m)
    return torch.max(log_Tensor, axis)[0] + \
           torch.log(torch.exp(log_Tensor - torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0], -1, 1)).sum(axis))


class IDCNN(nn.Module):
    """
      (idcnns): ModuleList(
    (0): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
    (1): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
    (2): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
    (3): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
  )
)
    """

    def __init__(self, input_size, filters, seq_len=256,kernel_size=3, num_block=4):
        super(IDCNN, self).__init__()
        self.filters = filters
        self.layers = [
            {"dilation": 1},
            {"dilation": 1},
            {"dilation": 2}]
        net = nn.Sequential()
        norms_1 = nn.ModuleList([LayerNorm(seq_len) for _ in range(len(self.layers))])
        norms_2 = nn.ModuleList([LayerNorm(seq_len) for _ in range(num_block)])
        for i in range(len(self.layers)):
            dilation = self.layers[i]["dilation"]
            single_block = nn.Conv1d(in_channels=filters,
                                     out_channels=filters,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     padding=kernel_size // 2 + dilation - 1)
            net.add_module("layer%d" % i, single_block)
            net.add_module("relu", nn.ReLU())
            net.add_module("layernorm", norms_1[i])

        self.linear = nn.Linear(input_size, filters)
        self.fc = nn.Linear(input_size, filters)
        self.idcnn = nn.Sequential()

        for i in range(num_block):
            self.idcnn.add_module("block%i" % i, net)
            self.idcnn.add_module("relu", nn.ReLU())
            self.idcnn.add_module("layernorm", norms_2[i])

    def forward(self, embeddings):
        embeddings = self.fc(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        output = self.idcnn(embeddings).permute(0, 2, 1)
        return output


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Bert_BiLSTM_CRF(nn.Module):
    def __init__(self, tag_to_ix, device, hidden_dim=500):
        super(Bert_BiLSTM_CRF, self).__init__()

        # cnn layer
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 158))
        # self.relu = nn.ReLU()
        self.idcnn = IDCNN(input_size=158, filters=200)

        # attention layer
        self.dropout = nn.Dropout(0.1)
        #

        self.batch_size = 1
        self.device = device
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        # self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=968, hidden_size=hidden_dim,
                            batch_first=True, dropout=0.1)

        # self.lstm1 = nn.LSTM(bidirectional=True, num_layers=2, input_size=158, hidden_size=100,
        #                     batch_first=True, dropout=0.1)
        self.transitions = nn.Parameter(torch.randn(
            self.tagset_size, self.tagset_size
        ))
        self.hidden_dim = hidden_dim
        self.start_label_id = self.tag_to_ix['[CLS]']
        self.end_label_id = self.tag_to_ix['[SEP]']
        self.fc = nn.Linear(hidden_dim * 2, self.tagset_size)
        self.bert1 = BertModel.from_pretrained('chinese_L-12_H-768_A-12')
        # self.bert2 = AlbertModel.from_pretrained('albert-large-zh')
        # self.bert.eval()
        self.transitions.data[self.start_label_id, :] = -10000
        self.transitions.data[:, self.end_label_id] = -10000
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.transitions.to(self.device)

    def init_hidden(self):
        return (torch.randn(4, self.batch_size, self.hidden_dim).to(self.device),
                torch.randn(4, self.batch_size, self.hidden_dim).to(self.device))

    def _forward_alg(self, feats):
        '''
        this also called alpha-recursion or forward recursion, to calculate log_prob of all barX
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # alpha_recursion,forward, alpha(zt)=p(zt,bar_x_1:t)
        log_alpha = torch.Tensor(batch_size, 1, self.tagset_size).fill_(-10000.).to(self.device)  # [batch_size, 1, 16]
        # normal_alpha_0 : alpha[0]=Ot[0]*self.PIs
        # self.start_label has all of the score. it is log,0 is p=1
        log_alpha[:, 0, self.start_label_id] = 0

        # feats: sentances -> word embedding -> lstm -> MLP -> feats
        # feats is the probability of emission, feat.shape=(1,tag_size)
        for t in range(1, T):
            log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)

        # log_prob of all barX
        log_prob_all_barX = log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    def _score_sentence(self, feats, label_ids):
        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size, self.tagset_size, self.tagset_size)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0], 1)).to(self.device)
        # the 0th node is start_label->start_word,the probability of them=1. so t begin with 1.
        for t in range(1, T):
            score = score + \
                    batch_transitions.gather(-1, (label_ids[:, t] * self.tagset_size + label_ids[:, t - 1]).view(-1, 1)) \
                    + feats[:, t].gather(-1, label_ids[:, t].view(-1, 1)).view(-1, 1)
        return score

    def _bert_enc(self, x, feature):
        """
        x: [batch_size, sent_len]
        enc: [batch_size, sent_len, 768]
        """
        # print("x.size:", x.size())
        with torch.no_grad():
            encoded_layer, _ = self.bert1(x)
            # encoded_layer, _ = self.bert2(x)
            # encoded_layer = self.bert2(x)
            # print(encoded_layer[0].size())
            # print(encoded_layer[1].size())
            # print("x:", x)
            # print("x:", x.size())
            # print("embedding:", len(self.bert(x)))
            # print("embedding: ", self.bert(x)[0][0].size())
            # print("encoded_layer:", len(encoded_layer[0][1]))
            # print("encoded_layer:", encoded_layer.size)
            # print("_:", type(_))

            # enc = encoded_layer[-1]
            enc = encoded_layer[0]
            # print("enc:", enc)
            # feature = feature.to(self.device)
            feature = self.idcnn(feature)
            # cnn = self.conv2(cnn)
            # print("lstm:", lstm.size())
            # print("enc:", enc.size())
            # print("feature:", feature.size())
            # print("before_enc:", enc.size())
            y = enc
            # print("y:", y.size())
            # print("feature:", feature.size())
            # print("batch_size:", y.size())
            n = []
            for i in range(x.size()[0]):
                z = torch.cat([y[i], feature[i]], dim=1)
                # print("z:", z.size())
                z = z.tolist()
                n.append(z)
            # m.append(n)
            n = torch.Tensor(n)
            n.to(self.device)
            # print("n:", n.size())
            # print("after_enc:", enc.size())
        return n

    def _viterbi_decode(self, feats):
        '''
        Max-Product Algorithm or viterbi algorithm, argmax(p(z_0:t|x_0:t))
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # batch_transitions=self.transitions.expand(batch_size,self.tagset_size,self.tagset_size)

        log_delta = torch.Tensor(batch_size, 1, self.tagset_size).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0.

        # psi is for the vaule of the last latent that make P(this_latent) maximum.
        psi = torch.zeros((batch_size, T, self.tagset_size), dtype=torch.long)  # psi[0]=0000 useless
        for t in range(1, T):
            # delta[t][k]=max_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # delta[t] is the max prob of the path from  z_t-1 to z_t[k]
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            # psi[t][k]=argmax_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # psi[t][k] is the path choosed from z_t-1 to z_t[k],the value is the z_state(is k) index of z_t-1
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = torch.zeros((batch_size, T), dtype=torch.long)

        # max p(z1:t,all_x|theta)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T - 2, -1, -1):
            # choose the state of z_t according the state choosed of z_t+1.
            path[:, t] = psi[:, t + 1].gather(-1, path[:, t + 1].view(-1, 1)).squeeze()

        return max_logLL_allz_allx, path

    def neg_log_likelihood(self, sentence, tags, feature):
        feats = self._get_lstm_features(sentence, feature)  # [batch_size, max_len, 16]
        # print("feats:", feats.size())
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return torch.mean(forward_score - gold_score)

    def _get_lstm_features(self, sentence, feature):
        """sentence is the ids"""
        # self.hidden = self.init_hidden()
        # self.idcnn = IDCNN(input_size=158, filters=200, seq_len=sentence.size()[-1])
        embeds = self._bert_enc(sentence, feature)  # [8, 75, 768]
        embeds = embeds.to(self.device)
        # print("embeds:", embeds.size())
        embeds = self.dropout(embeds)

        # 过lstm
        enc, _ = self.lstm(embeds)
        # print("enc:", enc.size())
        # 过attention
        query = self.dropout(enc)
        # query = enc
        # attn_output = self.attention(query)
        # print("attention:", attention)
        attn_output, alpha_n = self.attention_net(enc, query)
        # print("attn_output:", attn_output.size())
        attn_out = self.fc(attn_output)
        # print("attn_out:", attn_out.size())

        # lstm_feats = self.fc(enc)
        return attn_out  # [batch_szie, length, tagset_size]

    def attention_net(self, x, query, mask=None):
        d_k = query.size(-1)  # d_k为query的维度

        # query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len] print("query: ", query.shape,
        # x.transpose(1, 2).shape)  # torch.Size([128, 38, 128]) torch.Size([128, 128, 38]) 打分机制 scores: [batch,
        # seq_len, seq_len]
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        #         print("score: ", scores.shape)  # torch.Size([128, 38, 38])

        # 对最后一个维度 归一化得分
        alpha_n = F.softmax(scores, dim=-1)
        # print("alpha_n: ", alpha_n.size())
        # print("x:", x.size())
        # torch.Size([128, 38, 38]) 对权重化的x求和 [batch, seq_len, seq_len]·[batch,
        # seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]

        # context = torch.matmul(alpha_n, x).sum(1)
        context = torch.matmul(alpha_n, x)
        # print("context:", context.size())
        # print("x:", x)
        # print("context:", context)
        # print("context:", context)
        lstm = x * context
        # print("lstm:", x * context)

        return lstm, alpha_n

    def forward(self, sentence, feature):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence, feature)  # [8, 180,768]
        # print("lstm_feats:", lstm_feats.size())

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
