import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils import data
from torchvision import datasets, transforms
import os
from PIL import Image
import pandas as pd
import re
from konlpy.tag import Okt #pip install konlpy
from konlpy.tag import Komoran
#from konlpy.tag import Mecab
from tokenize import tokenize
from nltk import FreqDist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from soynlp import DoublespaceLineCorpus
from soynlp.tokenizer import RegexTokenizer




class CustomDataset(data.Dataset):
    def __init__(self, root, phase='train'):
        print("CustomDataset-> init")
        #count_vectorizer = make_vocab(root)
        self.root = root
        self.phase = phase
        self.labels = {}

        self.label_path = os.path.join(root, self.phase + '_hate.txt')
        with open(self.label_path, 'r',encoding="utf-8") as f:
            temp1 = []
            bias_list = []
            hate_list = []
            for line in f.readlines()[0:]:
                v = line.strip().split('\t')
                w = v[1]
                w = w.replace('!','')
                w = w.replace('.','')
                w = w.replace('^','')
                w = w.replace('♡','')
                w = w.replace('@','')
                w = w.replace('ㅎ','')
                w = w.replace('ㅉ','')
                w = w.replace('?','')
                w = w.replace('ㅜ','')
                w = w.replace('ㅠ','')
                w = w.replace('~','')
                w = w.replace('ㅋ','')
                w = w.replace('ㅡ','')
                w = w.replace('!','')
                w = w.replace('ㄷ','')
                w = w.replace('ㄹ','')
                w = w.replace('ㅇ','')
                w = w.replace(',','')
                w = w.replace('ㅈ','')
                w = w.replace('♥','')
                w = w.replace('ㅁ','')
                w = w.replace('ㅊ','')
                w = w.replace(';','')
                w = w.replace('ㄴ','')
                w = w.replace('ㆍ','')
                temp1.append(w)
                if phase != 'test':
                    bias_list.append(v[2])
                    hate_list.append(v[3])
        
        
        stopwords =['의','가','이','은','들','는','좀','잘',
                    '걍','과','도','를','으로','자','에','와','한','하다']
        
        comments_list = [] # 형태소로 자름
        
        okt = Okt()
        komoran =Komoran()
        tokenizer = RegexTokenizer()
        
        
        for sentence in temp1:
            temp_x =[]
            #temp_x= komoran.morphs(sentence,stem=True)
            temp_x= komoran.morphs(sentence)
            #temp_x = tokenizer.tokenize(sentence)
            temp_x = [word for word in temp_x if not word in stopwords]
            comments_list.append(temp_x) # 형태소로 잘리고
      
        
        vocab = FreqDist(np.hstack(comments_list)) #빈도수로 sort
        
        threshold = 2
        total_cnt = len(vocab)
        rare_cnt = 0
        total_freq = 0
        rare_freq = 0
        
        for key in vocab.keys():
            total_freq = total_freq + vocab[key] 
            if vocab[key] < threshold :
                rare_cnt = rare_cnt+1
                rare_freq = rare_freq + vocab[key]
                
        #         print('문장 집합(vocabulary)의 크기 :',total_cnt)
#         print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
#         print("문장 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
#         print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
        
        vocab_size = total_cnt - rare_cnt + 2
        vocab = vocab.most_common(vocab_size) 

        word_to_index = {word[0] : index + 2 for index, word in enumerate(vocab)}
        word_to_index['pad'] = 0
        word_to_index['unk'] = 0
        encoded = []
        
        for line in comments_list: 
            temp = []
            for w in line: 
                try:
                    temp.append(word_to_index[w])
                except KeyError: 
                    temp.append(word_to_index['unk']) # unk의 인덱스로 변환
            encoded.append(temp)
        #print(encoded[0:5])
#         rint(encoded.size())
    
        #max_len = max(len(length) for length in encoded)
        max_len = 74 # batch_size        
#         print("here")
#         print(a)
#         print("encoded")
#         print(len(encoded))
#         print('문장의최대 길이 : %d' % max_len)
#         print('문장의최소 최소 길이 : %d' % min(len(length) for length in encoded))
#         print('문장의 평균 길이 : %f' % (sum(map(len, encoded))/len(encoded)))
        
        for line in encoded:
            if len(line) < max_len: # 현재 샘플이 정해준 길이보다 짧으면
                line += [word_to_index['pad']] * (max_len - len(line))
                
        encoded = torch.LongTensor(encoded)
        #print(encoded[0:10])
        
        
     
        #encoded = pad_sequence(encoded,batch_first=True)
        #print(encoded.size)
        
#         print('패딩결과 최대 길이 : %d' % max(len(l) for l in encoded))
#         print('패딩결과의 최소 길이 : %d' % min(len(l) for l in encoded))
#         print('패딩결과의 평균 길이 : %f' % (sum(map(len, encoded))/len(encoded)))

        comments_vector = []
#         for comment in temp1:
#             comments_vector.append(count_vectorizer.transform([comment]).toarray()[0])
#         comments_vector = torch.FloatTensor(comments_vector)

        self.comments_vec = encoded  # 단어집합 숫자에 맞추고 pad, 한 결과 집합
        self.comments_list = temp1  # 문장 원본
                print(len(temp1))
        print(len(vocab))
        print(len(comments_list))
        print(len(encoded))
        
        if self.phase != 'test':
            bias_name_list = ['none', 'gender', 'others']
            hate_name_list = ['none', 'hate', 'offensive']
            from itertools import product
            bias_hate_list = [bias_name_list, hate_name_list]
            bias_hate_list = list(product(*bias_hate_list))
            label_list = []
            for idx in range(len(self.comments_list)):
                labels = (bias_list[idx], hate_list[idx])
                label_list.append(bias_hate_list.index(labels))
            self.label_list = label_list
            print(type(label_list))
            #print(type(label_list))
            #print(len(label_list)
    def __getitem__(self, index):
        if self.phase != 'test':
            return (self.comments_list[index], self.comments_vec[index]), self.label_list[index]
        elif self.phase == 'test':
            dummy = ""
            return (self.comments_list[index], self.comments_vec[index]), dummy

    def __len__(self):
        return len(self.comments_list)

def size_of_vocab():
    return 


def data_loader(root, phase='train', batch_size=16):
    print("CustomDataset-> data_loader")
    if phase == 'train':
        shuffle = True
    else:
        shuffle = False

    dataset = CustomDataset(root, phase)
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

