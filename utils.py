# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2019/11/07 22:11:33
@Author  :   Cao Shuai
@Version :   1.0
@Contact :   caoshuai@stu.scu.edu.cn
@License :   (C)Copyright 2018-2019, MILAB_SCU
@Desc    :   None
'''

import os
import numpy as np
import logging
import torch
from torch.utils.data import Dataset
from typing import Tuple, List
from pytorch_pretrained_bert import BertTokenizer

logger = logging.getLogger(__name__)

bert_model = 'chinese_L-12_H-768_A-12'
tokenizer = BertTokenizer.from_pretrained(bert_model)
# VOCAB = ('<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'B-LOC', 'B-ORG')

# 原始数据标签
# VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-INF', 'I-INF', 'B-PAT', 'I-PAT', 'B-OPS',
#          'I-OPS', 'B-DSE', 'I-DSE', 'B-DRG', 'I-DRG', 'B-LAB', 'I-LAB')
# VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-CHECK', 'I-CHECK', 'B-BODY', 'I-BODY',
#          'B-SIGNS', 'I-SIGNS', 'B-DISEASE', 'I-DISEASE', 'B-TREATMENT', 'I-TREATMENT')

# ccks2017标签
# VOCAB = ['<PAD>', '[CLS]', '[SEP]', 'O', 'B-symp', 'I-symp', 'B-body', 'I-body',
#          'B-dise', 'I-dise', 'B-chec', 'I-chec', 'B-cure', 'I-cure']

# ccks2018标签
# VOCAB = ['<PAD>', '[CLS]', '[SEP]', 'O', 'B-IndependentSymptoms', 'I-IndependentSymptoms', 'B-AnatomicSite',
#          'I-AnatomicSite', 'B-Operation', 'I-Operation', 'B-SymptomsDescribed', 'I-SymptomsDescribed',
#          'B-Medicine', 'I-Medicine']

# ccks2019标签
VOCAB = ['<PAD>', '[CLS]', '[SEP]', 'O',
         'B_解剖部位', 'I_解剖部位',
         'B_影像检查', 'I_影像检查',
         'B_疾病和诊断', 'I_疾病和诊断',
         'B_手术', 'I_手术',
         'B_药物', 'I_药物',
         'B_实验室检验', 'I_实验室检验']
# ccks2020标签
# VOCAB = ['<PAD>', '[CLS]', '[SEP]', 'O', 'B-POS', 'I-POS', 'B-LAB', 'I-LAB',
#          'B-SCR', 'I-SCR', 'B-DIS', 'I-DIS', 'B-MED', 'I-MED', 'B-OPE', 'I-OPE']

tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
print(tag2idx)
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
print(idx2tag)
MAX_LEN = 256

if 1 == 1.0:
    print(1)


class NerDataset(Dataset):
    # def __init__(self, f_path):
    #     with open(f_path, 'r', encoding='utf-8') as fr:
    #         entries = fr.read().strip().split('\n\n')
    #         # print("entries:", entries)
    #     sents, tags_li = [], []  # list of lists
    #     for entry in entries:
    #         # print("entry:", entry.splitlines())
    #         words = [line.split()[0] for line in entry.splitlines()]
    #         tags = ([line.split()[-1] for line in entry.splitlines()])
    #         assert len(words) == len(tags)
    #         # print("tags:", len(tags))
    #         # print("words:", len(words))
    #         if len(words) > MAX_LEN:
    #             # print(len(words))
    #             # 先对句号分段
    #             word, tag = [], []
    #             for char, t in zip(words, tags):
    #                 if char != '。':
    #                     if char != '\ue236':  # 测试集中有这个字符
    #                         word.append(char)
    #                         tag.append(t)
    #                 else:
    #                     sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
    #                     tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
    #                     word, tag = [], []
    #             # 最后的末尾
    #             if len(word):
    #                 sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
    #                 tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
    #                 word, tag = [], []
    #         else:
    #             # print(len(words))
    #             sents.append(["[CLS]"] + words + ["[SEP]"])
    #             tags_li.append(['[CLS]'] + tags + ['[SEP]'])
    #             while len(sents) < MAX_LEN:
    #                 sents.append(0)
    #                 tags_li.append(tag2idx['<PAD>'])
    #         for sent in sents:
    #             print("sent:", len(sent))
    #         # print("self.sents:", len(sents[0]))
    #         break
    #     # print("self.tags_li:", len(tags_li))
    #
    #     self.sents, self.tags_li = sents, tags_li

    def __init__(self, f_path):
        with open(f_path, 'r', encoding='utf-8') as fr:
            entries = fr.read().strip().split('\n\n')
            # print("entries:", entries)
        sents, tags_li = [], []  # list of lists

        for entry in entries:
            # print("entry:", entry.splitlines())
            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            assert len(words) == len(tags)

            labels = words
            sentences = tags

            # print("tags:", len(tags))
            # print("words:", len(words))

            # if len(words) == 1576:
            #     print("words:", words)
            #     max = len(words)

            if len(words) > MAX_LEN - 2:
                # print("words:", words)
                # print("sents:", len(sents))
                n = float(len(words)) / float(MAX_LEN - 2)
                # print("n:", n)
                if int(n) == n:
                    n = int(n)
                    # print("n1:", n)
                    for j in range(n):
                        sentences = words[j * (MAX_LEN - 2): (j + 1) * (MAX_LEN - 2)]
                        labels = tags[j * (MAX_LEN - 2): (j + 1) * (MAX_LEN - 2)]
                        sentences = ['[CLS]'] + sentences + ['[SEP]']
                        labels = ["[CLS]"] + labels + ['[SEP]']

                        # print("sentences1:", len(sentences))
                        # print("labels1:", len(labels))
                        # if len(sentences) != 256:
                        #     print("sentences1:", sentences)
                        #     print("labels1:", labels)
                        sents.append(sentences)
                        tags_li.append(labels)
                else:
                    n = int(n) + 1
                    # print("n2:", n)
                    for j in range(n):
                        if j != n - 1:
                            # print("j1:", j)
                            sentences = words[j * (MAX_LEN - 2): (j + 1) * (MAX_LEN - 2)]
                            # print("sentences1:", sentences)
                            labels = tags[j * (MAX_LEN - 2): (j + 1) * (MAX_LEN - 2)]
                            sentences = ['[CLS]'] + sentences + ['[SEP]']
                            labels = ["[CLS]"] + labels + ['[SEP]']

                            # print("sentences2:", len(sentences))
                            # print("labels2:", len(labels))
                            # if len(sentences) != 256:
                            #     print("sentences2:", sentences)
                            #     print("labels2:", labels)
                            sents.append(sentences)
                            tags_li.append(labels)
                        else:
                            # print("j2:", j)
                            sentences = words[j * (MAX_LEN - 2):]
                            # print("sentences2:", sentences)
                            labels = tags[j * (MAX_LEN - 2):]
                            while len(sentences) < MAX_LEN - 2:
                                sentences.append('[PAD]')
                                labels.append('<PAD>')
                            sentences = ['[CLS]'] + sentences + ['[SEP]']
                            labels = ["[CLS]"] + labels + ['[SEP]']

                            # print("sentences3:", len(sentences))
                            # print("labels3:", len(labels))
                            # if len(sentences) != 256:
                            #     print("sentences3:", sentences)
                            #     print("labels3:", labels)
                            sents.append(sentences)
                            tags_li.append(labels)
                # print("sents:", len(sents))
            else:
                sentences = words
                labels = tags
                while len(sentences) < MAX_LEN - 2:
                    sentences.append('[PAD]')
                    labels.append('<PAD>')
                sentences = ['[CLS]'] + sentences + ['[SEP]']
                labels = ["[CLS]"] + labels + ['[SEP]']

                # print("sentences4:", len(sentences))
                # print("labels4:", len(labels))
                # if len(sentences) != 256:
                #     print("sentences4:", sentences)
                #     print("labels4:", labels)
                sents.append(sentences)
                tags_li.append(labels)
        # print("sents:", len(sents))
        # print("self.tags_li:", len(tags_li))

        self.sents, self.tags_li = sents, tags_li

    # 键值对映射
    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]
        # print("words:", words)
        # print("len(words):", len(words))
        x, y = [], []
        is_heads = []
        # print("words:", len(words))
        # print("tags:", len(tags))
        # print("words:", words)
        # print("tags:", tags)
        for w, t in zip(words, tags):
            # print("w:{0}, t:{1}".format(w, t))
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]

            # print("tokens:", tokens)
            xx = tokenizer.convert_tokens_to_ids(tokens)
            # print("tokens:{0}, xx:{1}".format(tokens, xx))
            # assert len(tokens) == len(xx), f"len(tokens)={len(tokens)}, len(xx)={len(xx)}"

            # 中文没有英文wordpiece后分成几块的情况
            is_head = [1] + [0] * (len(tokens) - 1)
            t = [t] + ['<PAD>'] * (len(tokens) - 1)
            yy = [tag2idx[each] for each in t]  # (T,)
            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)
            # print("xx:", len(xx))
            # print("yy:", len(yy))

        # while len(x) < MAX_LEN:
        #     x.append(0)
        #     is_heads.append(0)
        #     y.append(tag2idx['<PAD>'])
        # print("len(x):", len(x))

        assert len(x) == len(y) == len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        # for i in range(len(x)):
        #     if i < len(words.split()):
        #         print("words:{0} x:{1}".format(words.split()[i], x[i]))
        #     else:
        #         print("words:  x:{0}".format(x[i]))
        return words, x, is_heads, tags, y, seqlen

    def __len__(self):
        return len(self.sents)


def pad(batch):
    """Pads to the longest sample"""
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)

    f = torch.LongTensor

    return words, f(x), is_heads, tags, f(y), seqlens
