#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable



class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()

        #三个待输入的数据
        self.embedding = nn.Embedding(5000, 64)
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, bidirectional=True)
        #self.rnn = nn.GRU(input_size= 64, hidden_size= 128, num_layers= 2, bidirectional= True)
        self.f1 = nn.Sequential(nn.Linear(256, 10),
                                F.softmax(input()))


    def forward(self, x):
        x = self.embedding(x)  # batch_size x seq_len x embedding_size 64*600*64
        x = x.permute(1, 0, 2) # seq_len x batch_size x embedding_size 600*64*64

        # x为600*64*256, h_n为2*64*128 lstm_out
        # Sentence_length * Batch_size * (hidden_layers * 2 [bio-direct]) h_n
        # （num_layers * 2） * Batch_size * hidden_layers
        x, (h_n, c_n) = self.rnn(x)
        final_feature_map = F.dropout(h_n, 0.8)
        feature_map = torch.cat([final_feature_map[i, :, :] for i in range(final_feature_map.shape[0])], dim = 1)
        final_out = self.f1(feature_map)
        return final_out

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(5000, 64)
        self.conv = nn.Sequential(nn.Conv1d(in_channels= 64,
                                            out_channels= 256,
                                            kernel_size= 5),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size= 596))

        self.f1 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.embedding(x) # batch_size x seq_len x embedding_size 64*600*64
        x = x.permute(0, 2, 1) #64*64*600 变成 bs x embs x seq_len
        x = self.conv(x) #Conv1后64*256*596,ReLU后不变,NaxPool1d后64*256*1

        x = x.view(-1, x.size(1)) # 64 * 256
        x = F.dropout(x, 0.8)
        x = self.f1(x)  # 64 x 10  bs x class_num
        return x


if __name__ == '__main__':
    net = TextRNN()
    print(net)