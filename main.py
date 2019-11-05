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
txt_char = poetry_corpus[:11]
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

use_gpu = True



class CharRNN(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_size, num_layers, dropout):
        super(RNNmodel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # num_classes个词,纬度是embed_dim
        self.word_to_vec = nn.Embedding(num_classes, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers, dropout)
        self.project = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, hs = None):
        batch = x.shape[0]
        if hs is None:
            hs = Variable(torch.zeros(self.num_layers, batch, self.hidden_size))
            if use_gpu:
                hs = hs.cuda()
        word_embed = self.word_to_vec(x)
        word_embed= word_embed.permute(1, 0, 2)
        out, h0 =self.rnn(word_embed, hs)
        le, mb, hd = out.shape
        out = out.view(le * mb, hd)
        out = self.project(out)
        out = out.view(le, mb, -1)
        out = out.permute(1, 0, 2).contiguous()
        return out.view(-1, out.shape[2]), h0

class ScheduledOptim(object):
    """A wrapper class for learning rate scheduling
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lr = self.optimizer.param_groups[0]['lr']
        self.current_steps = 0

    def step(self):
        "Step by the inner optimizer"
        self.current_steps += 1
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def lr_multi(self, multi):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= multi
        self.lr = self.optimizer.param_groups[0]['lr']

    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @property
    def learning_rate(self):
        return self.lr
    
# 训练模型
from torch.utils.data import DataLoader
batch_size = 128
train_data = DataLoader(train_set, batch_size, True, num_workers = 4)


model = CharRNN(convert.vocab_size, 512, 512, 2, 0.5)
if use_gpu:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
basic_optimizer = torch.optim.Adam(model.parameters(), lr = le-3)
optimizer = ScheduledOptim(basic_optimizer)

epoch = 20
for e in range(epoches):
    train_loss = 0
    for data in train_data:
        x, y = data
        y = y.long()
        if use_gpu:
            x = x.cuda()
            y = y.cuda()
        x, y = Variable(x), Variable(y)
        
        # Forward
        score, _ = model(x)
        loss = criterion(score, y.view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradient
        nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()
        
        train_loss += loss.data[0]
    print('epoch: {}, perplexity is: {:.3f}, lr:{:.1e}'.format(e+1, np.exp(train_loss / len(train_data)), optimizer.lr))

import pickle
torch.save(model_object, 'model.pkl')

model = torch.load('model.pkl')

# 生成文本,默认选择概率最高的5个，随机选择一个作为输出，剩下来的传递到下一个cell; torch.topk
def pick_top_n(preds, top_n=5):
    top_pred_prob, top_pred_label = torch.topk(preds, top_n, 1)
    top_pred_prob /= torch.sum(top_pred_prob)
    top_pred_prob = top_pred_prob.squeeze(0).cpu().numpy()
    top_pred_label = top_pred_label.squeeze(0).cpu().numpy()
    c = np.random.choice(top_pred_label, size = 1, p = top_pred_prob)
    return c

begin = '天青色等烟雨'
text_len = 30

model = model.eval()
samples = [convert.word_to_int(c) for c in begin]
input_txt = torch.LongTensor(samples)[None]
if use_gpu:
    input_txt = input_txt.cuda()
input_txt = Variable(input_txt)
_, init_state = model(input_txt)
result = samples
model_input = input_txt[:, -1][:, None]
for i in range(text_len):
    out, init_state = model(model_input, init_state)
    pred = pick_top_n(out.data)
    model_input = Variable(torch.LongTensor(pred))[None]
    if use_gpu:
        model_input = model_input.cuda()
    result.append(pred[0])
text = convert.arr_to_text(result)
print('Generate text is: {}'.format(text))    
        

        
    
     
        
        
        