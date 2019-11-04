#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:21:18 2019

@author: daniel
"""
class TextConverter(object):
    def __init__(self, text_path, max_vocab = 5000):
        """建立字符转换器，将文本转换为数字，对所有非重复的字符，可以从0开始建立索引
        Args:
            text_path: 文本的位置
            max_vocab: 最大的单词数量, default = 5000
        """
        # 将空格，逗号，等其它字符替换为空格
        with open(text_path, 'r') as f:
            text = f.read()
        text = text.replace('\n', ' ').replace('\r', ' ').replace(',', ' ').replace('。', '').replace('，', ' ')
        
        # 去掉重复的字符,得到词汇表 vocab
        vocab = set(text)
        
        # 如果单词总数超过最大数值，则去掉频率最低的
        vocab_count = {}
        
        #计算频率并排序
            #初始化,每个词的数目设置0
        for word in vocab:
            vocab_count[word] = 0
            #统计每个词的数目,发现一个自加1
        for word in text:
            vocab_count[word] += 1
        #初始化一个list,方便后边排序运算
        vocab_count_list = []
        for word in vocab_count:
            vocab_count_list.append((word, vocab_count[word]))
        # 排序    
        vocab_count_list.sort(key=lambda x: x[1], reverse=True)
            
        """        
        # list 格式[('鹤', 708), ('漾', 77), ('畋', 6), ...], 不能直接用 vocab_count_list['汪']找字符
        """
        if len(vocab_count_list) > max_vocab:
            vocab_count_list = vocab_count_list[:max_vocab]
        #抽取list中第一列到词汇表vocab中,并附到self.vocab中给这个类中公用
        vocab = [x[0] for x in vocab_count_list]
        self.vocab = vocab
        
        # 建立索引字典
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))
        
    

    def vocab_size(self):
        return len(self.vocab)
    
    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)
    
    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')
    
    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)
    
    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

class TextDataset(object):
    """
    构建数据集
    args:
        arr: 传入的序列矩阵
        x: 对应一个分割
        y: 与x对应,平移一位
    """
    def __init__(self, arr):
        self.arr = arr
    def __getitem__(self, item):
        # 对每一个分割最一个平移
        x = self.arr[item,:]
        # 初始化 y
        y = torch.zeros(x.shape)
        # 将x的第2个直到最后，赋值给y的前n-1个; x的第一个赋值给y的最后一个
        y[:-1], y[-1] = x[1:], x[0]
        return x, y
    def __len__(self):
        return self.arr.shape[0]


          
    
convert = TextConverter('./dataset/poetry.txt', max_vocab=10000)
n_step =20
num_seq = int(len(poetry_corpus)/n_step)

text = poetry_corpus[:num_seq * n_step]   

import torch

arr = convert.text_to_arr(text)

arr = arr.reshape((num_seq, -1))

arr = torch.from_numpy(arr)

train_set = TextDataset(arr)

x, y = train_set[0]

"""
建立模型,第一层是嵌入层,第二层是RNN层,第三层是线性层(因为是分类问题),最后输出预测的字符
"""
from torch import nn
from torch.autograd import Variable

use_gpu = Ture

class CharRNN(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_size, num_layers, dropout):
        super("tangRNN", self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.word_to_vec = nn.Embedding(num_classes, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_size, num_layers, dropout)
        self.project = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, hs = None):
        batch = x.shape
     
        
        
        