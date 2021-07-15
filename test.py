# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import argparse
from torch.utils import data
from model import Net
from crf import Bert_BiLSTM_CRF
from utils import NerDataset, pad, VOCAB, tokenizer, tag2idx, idx2tag
from x import process
from utils import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

print(torch.__version__)

print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
print(torch.cuda.current_device())

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

with open("Chinese.txt", "r", encoding="UTF-8") as f:
    chinese = []
    sort = []
    answer = {}
    contents = f.readlines()
    for content in contents:
        txt = content.split()
        chinese.append(txt[0])
        sort.append(txt[1])
        answer[txt[0]] = txt[1]

data_year = 2019
if data_year == 2020:
    t_begin = {4: 'B-POS', 6: 'B-LAB', 8: 'B-DIS', 10: 'B-SCR', 12: 'B-MED', 14: 'B-OPE'}
    # 解刨部位    影像检查      疾病         手术          药物         手术
elif data_year == 2019:
    t_begin = {4: 'B_解剖部位', 6: 'B_影像检查', 8: 'B_疾病和诊断', 10: 'B_手术', 12: 'B_药物', 14: 'B_实验室检验'}
elif data_year == 2018:
    t_begin = {4: 'B-IndependentSymptoms', 6: 'B-AnatomicSite', 8: 'B-Operation', 10: 'B-SymptomsDescribed',
               12: 'B-Medicine'}
    # 独立症状 解刨部位 手术 症状描述 药物
elif data_year == 2017:
    t_begin = {4: 'B-symp', 6: 'B-body', 8: 'B-dise', 10: 'B-chec', 12: 'B-cure'}
    # 症状体征 身体部位 疾病诊断 检查检验  治疗


def train(model, iterator, optimizer, criterion, device):
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    g = 0
    all = 0
    print("data_year:", data_year)
    print("t_begin:", t_begin)
    for i, batch in enumerate(iterator):
        # print("i:{0} batch:{1}".format(i, len(batch)))
        words, x, is_heads, tags, y, seqlens = batch
        # sentence, masks, tags = batch
        # print("words:", len(words[0].split()))
        # print("words:", words)
        # print("x:", x.size())
        # print("y:", y.size())
        # print("x:", x)
        # print("y:", y)
        # print("words:", words)
        # print("batch:", len(tags))
        # print("sentence:", tags[i])
        for j in range(len(tags)):
            for tag in tags[j].split(" "):
                if data_year == 2018:
                    if tag == 'B-IndependentSymptoms':
                        a += 1
                    elif tag == 'B-AnatomicSite':
                        b += 1
                    elif tag == 'B-Operation':
                        c += 1
                    elif tag == 'B-SymptomsDescribed':
                        d += 1
                    elif tag == 'B-Medicine':
                        e += 1
                    all += 1
                elif data_year == 2019:
                    if tag == 'B_解剖部位':
                        a += 1
                    elif tag == 'B_影像检查':
                        b += 1
                    elif tag == 'B_疾病和诊断':
                        c += 1
                    elif tag == 'B_手术':
                        d += 1
                    elif tag == 'B_药物':
                        e += 1
                    elif tag == 'B_实验室检验':
                        g += 1
                    all += 1

        # print("batch:",batch)
        x = x.to(device)
        y = y.to(device)
        # x = x.cuda()
        # y = y.cuda()
        # print("words:", len(words[0].split()))
        # print("words:", len(words))
        # print("x:", x.size())
        z = process(words)
        # print("z:", len(z))
        # for z1 in z:
        #     print("z1:", len(z1))
        #     for z2 in z1:
        #         print("z2:", z2)

        z = torch.Tensor(z)
        z = z.to(device)
        model.train()
        # _y = y  # for monitoring

        # logits = model(x, z)[1].float().to(device)
        # print("logits:", logits.type())
        # print("y:", y.size())
        # target = y.squeeze()
        # print("target:", target.size())
        # loss = criterion(logits, target)

        loss = model.neg_log_likelihood(x, y, z)  # logits: (N, T, VOCAB), y: (N, T)
        optimizer.zero_grad()
        # loss = model.neg_log_likelihood(x, y, z)  # logits: (N, T, VOCAB), y: (N, T)

        loss.backward()

        optimizer.step()

        # if i == 0:
        #     print("=====sanity check======")
        #     print("words:", words[0])
        #     print("x:", x.cpu().numpy()[0][:seqlens[0]])
        #     print("tokens:", tokenizer.convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
        #     print("is_heads:", is_heads[0])
        #     print("y:", _y.cpu().numpy()[0][:seqlens[0]])
        #     print("tags:", tags[0])
        #     print("seqlen:", seqlens[0])
        #     print("=======================")
        if i == 0:
            print("=====   training   ======")
        if i % 10 == 0:  # monitoring
            print(f"step: {i}, loss: {loss.item()}")

    if data_year == 2018:
        print("---training data---")
        print("独立症状:", a)
        print("解刨部位:", b)
        print("手术:", c)
        print("症状描述:", d)
        print("药物:", e)
        print("ALL:", a + b + c + d + e)
        print("total:", all)
    elif data_year == 2019:
        print("解剖部位:", a)
        print("影像检查:", b)
        print("疾病和诊断:", c)
        print("手术:", d)
        print("药物:", e)
        print("实验室检验:", g)
        print("ALL:", a + b + c + d + e + g)
        print("total:", all)


def eval(model, iterator, f, device):
    model.eval()
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    g = 0
    all = 0
    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch
            for j in range(len(tags)):
                for tag in tags[j].split(" "):
                    if data_year == 2018:
                        if tag == 'B-IndependentSymptoms':
                            a += 1
                        elif tag == 'B-AnatomicSite':
                            b += 1
                        elif tag == 'B-Operation':
                            c += 1
                        elif tag == 'B-SymptomsDescribed':
                            d += 1
                        elif tag == 'B-Medicine':
                            e += 1
                        all += 1
                    elif data_year == 2019:
                        if tag == 'B_解剖部位':
                            a += 1
                        elif tag == 'B_影像检查':
                            b += 1
                        elif tag == 'B_疾病和诊断':
                            c += 1
                        elif tag == 'B_手术':
                            d += 1
                        elif tag == 'B_药物':
                            e += 1
                        elif tag == 'B_实验室检验':
                            g += 1
                        all += 1

            x = x.to(device)
            # y = y.to(device)
            z = process(words)
            z = torch.Tensor(z)
            z = z.to(device)
            _, y_hat = model(x, z)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

        if data_year == 2018:
            print("---eval data---")
            print("独立症状:", a)
            print("解刨部位:", b)
            print("手术:", c)
            print("症状描述:", d)
            print("药物:", e)
            print("ALL:", a + b + c + d + e)
            print("total:", all)
        elif data_year == 2019:
            print("解剖部位:", a)
            print("影像检查:", b)
            print("疾病和诊断:", c)
            print("手术:", d)
            print("药物:", e)
            print("实验室检验:", g)
            print("ALL:", a + b + c + d + e + g)
            print("total:", all)

    ## gets results and save
    with open("temp", 'w', encoding='utf-8') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            # print("{0} {1} {2}".format(len(preds), len(words.split()), len(tags.split())))
            assert len(preds) == len(words.split()) == len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
                # print("{0}\t{1}\t{2}".format(w, t, p))
            fout.write("\n")

    ## calc metric
    y_true = np.array(
        [tag2idx[line.split()[1]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])
    y_pred = np.array(
        [tag2idx[line.split()[2]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])

    check_true = 0
    check_pred = 0
    check_tp = 0
    body_true = 0
    body_pred = 0
    body_tp = 0
    signs_true = 0
    signs_pred = 0
    signs_tp = 0
    disease_true = 0
    disease_pred = 0
    disease_tp = 0
    treatment_true = 0
    treatment_pred = 0
    treatment_tp = 0
    medical_true = 0
    medical_pred = 0
    medical_tp = 0

    true_begin = []
    true_len = []
    pred_begin = []
    pred_len = []

    for i in range(len(y_pred)):
        if i != len(y_pred) - 1:
            if y_pred[i] in t_begin.keys():
                pred_begin.append(i)
                if y_pred[i + 1] != y_pred[i] + 1:
                    pred_len.append(1)
                else:
                    j = i + 1
                    while y_pred[j] == y_pred[j + 1]:
                        j += 1
                        # print("j:", j)
                        if j + 1 >= len(y_pred):
                            break
                    pred_len.append(j - i)
        else:
            if y_pred[i] in t_begin.keys():
                pred_begin.append(i)
                pred_len.append(1)

    for i in range(len(y_true)):
        if i != len(y_true) - 1:
            if y_true[i] in t_begin.keys():
                true_begin.append(i)
                if y_true[i + 1] != y_true[i] + 1:
                    true_len.append(1)
                else:
                    j = i + 1
                    while y_true[j] == y_true[j + 1]:
                        j += 1
                        if j + 1 >= len(y_true):
                            break
                    true_len.append(j - i)
        else:
            if y_true[i] in t_begin.keys():
                true_begin.append(i)
                true_len.append(1)
    assert len(y_pred) == len(y_true)

    true_dict = {}
    pred_dict = {}
    for i in range(len(true_begin)):
        true_dict[true_begin[i]] = true_len[i]
    for i in range(len(pred_begin)):
        pred_dict[pred_begin[i]] = pred_len[i]

    tp = 0
    fp = 0
    fn = 0
    for i, j in true_dict.items():
        # print("begin:{0} length:{1}".format(i, j))
        if i in pred_dict.keys() and j in pred_dict.values() and j == pred_dict[i]:
            # print("begin:{0} length:{1}".format(i, j))
            if data_year == 2017 or data_year == 2018:
                if y_true[i] == 4 and y_pred[i] == 4:
                    signs_tp += 1
                elif y_true[i] == 6 and y_pred[i] == 6:
                    body_tp += 1
                elif y_true[i] == 8 and y_pred[i] == 8:
                    disease_tp += 1
                elif y_true[i] == 10 and y_pred[i] == 10:
                    check_tp += 1
                elif y_true[i] == 12 and y_pred[i] == 12:
                    treatment_tp += 1
                else:
                    continue
            elif data_year == 2019 or data_year == 2020:
                if y_true[i] == 4 and y_pred[i] == 4:
                    body_tp += 1
                elif y_true[i] == 6 and y_pred[i] == 6:
                    check_tp += 1
                elif y_true[i] == 8 and y_pred[i] == 8:
                    disease_tp += 1
                elif y_true[i] == 10 and y_pred[i] == 10:
                    treatment_tp += 1
                elif y_true[i] == 12 and y_pred[i] == 12:
                    medical_tp += 1
                elif y_true[i] == 14 and y_pred[i] == 14:
                    signs_tp += 1
                else:
                    continue
            if y_true[i] == y_pred[i]:
                tp += 1
        elif i in pred_dict.keys() and j != pred_dict[i]:
            fn += 1
        elif i not in pred_dict.keys():
            fn += 1

    for i, j in pred_dict.items():
        if i not in true_dict.keys():
            fp += 1

    for begin in true_begin:
        if data_year == 2017 or data_year == 2018:
            if y_true[begin] == 4:
                signs_true += 1
            elif y_true[begin] == 6:
                body_true += 1
            elif y_true[begin] == 8:
                disease_true += 1
            elif y_true[begin] == 10:
                check_true += 1
            elif y_true[begin] == 12:
                treatment_true += 1
            else:
                print("ERROR2 locate:{0} {1}".format(begin, y_true[begin]))
        elif data_year == 2019 or data_year == 2020:
            if y_true[begin] == 4:
                body_true += 1
            elif y_true[begin] == 6:
                check_true += 1
            elif y_true[begin] == 8:
                disease_true += 1
            elif y_true[begin] == 10:
                treatment_true += 1
            elif y_true[begin] == 12:
                medical_true += 1
            elif y_true[begin] == 14:
                signs_true += 1
            else:
                print("ERROR2 locate:{0} {1}".format(begin, y_true[begin]))

    for begin in pred_begin:
        # print("{0} {1}".format(begin, y_pred[begin]))
        if data_year == 2017 or data_year == 2018:
            if y_pred[begin] == 4:
                signs_pred += 1
            elif y_pred[begin] == 6:
                body_pred += 1
            elif y_pred[begin] == 8:
                disease_pred += 1
            elif y_pred[begin] == 10:
                check_pred += 1
            elif y_pred[begin] == 12:
                treatment_pred += 1
            else:
                print("ERROR2 locate:{0} {1}".format(begin, y_pred[begin]))
        elif data_year == 2019 or data_year == 2020:
            if y_pred[begin] == 4:
                body_pred += 1
            elif y_pred[begin] == 6:
                check_pred += 1
            elif y_pred[begin] == 8:
                disease_pred += 1
            elif y_pred[begin] == 10:
                treatment_pred += 1
            elif y_pred[begin] == 12:
                medical_pred += 1
            elif y_pred[begin] == 14:
                signs_pred += 1
            else:
                print("ERROR2 locate:{0} {1}".format(begin, y_pred[begin]))

    if data_year == 2017 or data_year == 2018:
        check_gold = check_true
        check_predict = check_pred
        check_correct = check_tp
        if check_predict == 0:
            check_precision = 0.
        else:
            check_precision = float(check_correct) / float(check_predict)
        if check_gold == 0:
            check_recall = 0.
        else:
            check_recall = float(check_correct) / float(check_gold)
        if check_precision == 0. or check_recall == 0.:
            check_F_score = 0.
        else:
            check_F_score = 2 * check_precision * check_recall / (check_precision + check_recall)

        body_gold = body_true
        body_predict = body_pred
        body_correct = body_tp
        if body_predict == 0:
            body_precision = 0.
        else:
            body_precision = float(body_correct) / float(body_predict)
        if body_gold == 0:
            body_recall = 0.
        else:
            body_recall = float(body_correct) / float(body_gold)
        if body_precision == 0. or body_recall == 0.:
            body_F_score = 0.
        else:
            body_F_score = 2 * body_precision * body_recall / (body_precision + body_recall)

        signs_gold = signs_true
        signs_predict = signs_pred
        signs_correct = signs_tp
        if signs_predict == 0:
            signs_precision = 0.
        else:
            signs_precision = float(signs_correct) / float(signs_predict)
        if signs_gold == 0:
            signs_recall = 0.
        else:
            signs_recall = float(signs_correct) / float(signs_gold)
        if signs_precision == 0. or signs_recall == 0.:
            signs_F_score = 0.
        else:
            signs_F_score = 2 * signs_precision * signs_recall / (signs_precision + signs_recall)

        disease_gold = disease_true
        disease_predict = disease_pred
        disease_correct = disease_tp
        if disease_predict == 0:
            disease_precision = 0.
        else:
            disease_precision = float(disease_correct) / float(disease_predict)
        if disease_gold == 0:
            disease_recall = 0.
        else:
            disease_recall = float(disease_correct) / float(disease_gold)
        if disease_precision == 0. or disease_recall == 0.:
            disease_F_score = 0.
        else:
            disease_F_score = 2 * disease_precision * disease_recall / (disease_precision + disease_recall)

        treatment_gold = treatment_true
        treatment_predict = treatment_pred
        treatment_correct = treatment_tp
        if treatment_predict == 0:
            treatment_precision = 0.
        else:
            treatment_precision = float(treatment_correct) / float(treatment_predict)
        if treatment_gold == 0:
            treatment_recall = 0.
        else:
            treatment_recall = float(treatment_correct) / float(treatment_gold)
        if treatment_precision == 0. or treatment_recall == 0.:
            treatment_F_score = 0.
        else:
            treatment_F_score = 2 * treatment_precision * treatment_recall / (treatment_precision + treatment_recall)

    elif data_year == 2019 or data_year == 2020:
        check_gold = check_true
        check_predict = check_pred
        check_correct = check_tp
        if check_predict == 0:
            check_precision = 0.
        else:
            check_precision = float(check_correct) / float(check_predict)
        if check_gold == 0:
            check_recall = 0.
        else:
            check_recall = float(check_correct) / float(check_gold)
        if check_precision == 0. or check_recall == 0.:
            check_F_score = 0.
        else:
            check_F_score = 2 * check_precision * check_recall / (check_precision + check_recall)

        body_gold = body_true
        body_predict = body_pred
        body_correct = body_tp
        if body_predict == 0:
            body_precision = 0.
        else:
            body_precision = float(body_correct) / float(body_predict)
        if body_gold == 0:
            body_recall = 0.
        else:
            body_recall = float(body_correct) / float(body_gold)
        if body_precision == 0. or body_recall == 0.:
            body_F_score = 0.
        else:
            body_F_score = 2 * body_precision * body_recall / (body_precision + body_recall)

        signs_gold = signs_true
        signs_predict = signs_pred
        signs_correct = signs_tp
        if signs_predict == 0:
            signs_precision = 0.
        else:
            signs_precision = float(signs_correct) / float(signs_predict)
        if signs_gold == 0:
            signs_recall = 0.
        else:
            signs_recall = float(signs_correct) / float(signs_gold)
        if signs_precision == 0. or signs_recall == 0.:
            signs_F_score = 0.
        else:
            signs_F_score = 2 * signs_precision * signs_recall / (signs_precision + signs_recall)

        disease_gold = disease_true
        disease_predict = disease_pred
        disease_correct = disease_tp
        if disease_predict == 0:
            disease_precision = 0.
        else:
            disease_precision = float(disease_correct) / float(disease_predict)
        if disease_gold == 0:
            disease_recall = 0.
        else:
            disease_recall = float(disease_correct) / float(disease_gold)
        if disease_precision == 0. or disease_recall == 0.:
            disease_F_score = 0.
        else:
            disease_F_score = 2 * disease_precision * disease_recall / (disease_precision + disease_recall)

        treatment_gold = treatment_true
        treatment_predict = treatment_pred
        treatment_correct = treatment_tp
        if treatment_predict == 0:
            treatment_precision = 0.
        else:
            treatment_precision = float(treatment_correct) / float(treatment_predict)
        if treatment_gold == 0:
            treatment_recall = 0.
        else:
            treatment_recall = float(treatment_correct) / float(treatment_gold)
        if treatment_precision == 0. or treatment_recall == 0.:
            treatment_F_score = 0.
        else:
            treatment_F_score = 2 * treatment_precision * treatment_recall / (treatment_precision + treatment_recall)

        medical_gold = medical_true
        medical_predict = medical_pred
        medical_correct = medical_tp
        if medical_predict == 0:
            medical_precision = 0.
        else:
            medical_precision = float(medical_correct) / float(medical_predict)
        if medical_gold == 0:
            medical_recall = 0.
        else:
            medical_recall = float(medical_correct) / float(medical_gold)
        if medical_precision == 0. or medical_recall == 0.:
            medical_F_score = 0.
        else:
            medical_F_score = 2 * medical_precision * medical_recall / (medical_precision + medical_recall)

    gold_num = len(true_begin)
    predict_num = len(pred_begin)
    correct_num = tp

    # 验证

    # print("y_true:", len(y_true))
    # print("y_pred:", len(y_pred))
    # print("true_begin:", len(true_begin))
    # print("true_len:", len(true_len))
    # print("pred_begin:", len(pred_begin))
    # print("pred_len:", len(pred_len))
    if data_year == 2017 or data_year == 2018:
        assert signs_tp <= signs_pred or body_tp <= body_pred or disease_tp <= disease_pred or check_tp <= check_pred or treatment_tp <= treatment_pred
        assert signs_tp + body_tp + disease_tp + check_tp + treatment_tp == correct_num
        assert signs_true + body_true + disease_true + check_true + treatment_true == gold_num
        assert signs_pred + body_pred + disease_pred + check_pred + treatment_pred == predict_num
    elif data_year == 2019 or data_year == 2020:
        assert signs_tp <= signs_pred or body_tp <= body_pred or disease_tp <= disease_pred or check_tp <= check_pred or treatment_tp <= treatment_pred or medical_tp <= medical_pred
        assert signs_tp + body_tp + disease_tp + check_tp + treatment_tp + medical_tp == correct_num
        assert signs_true + body_true + disease_true + check_true + treatment_true + medical_true == gold_num
        assert signs_pred + body_pred + disease_pred + check_pred + treatment_pred + medical_pred == predict_num

    if predict_num == 0:
        Precision = 0.
    else:
        Precision = float(correct_num) / float(predict_num)
    if gold_num == 0:
        Recall = 0.
    else:
        Recall = float(correct_num) / float(gold_num)
    if Precision == 0. and Recall == 0.:
        F_score = 0.
    else:
        F_score = 2 * Precision * Recall / (Precision + Recall)
    print("\t\tPrecision  Recall\tF-score")
    if data_year == 2017:
        print("症状体征:\t\t%.4f\t%.4f\t%.4f" % (signs_precision, signs_recall, signs_F_score))
        print("身体部位:\t\t%.4f\t%.4f\t%.4f" % (body_precision, body_recall, body_F_score))
        print("疾病诊断:\t\t%.4f\t%.4f\t%.4f" % (disease_precision, disease_recall, disease_F_score))
        print("检查检验:\t\t%.4f\t%.4f\t%.4f" % (check_precision, check_recall, check_F_score))
        print("治疗:\t\t%.4f\t%.4f\t%.4f" % (treatment_precision, treatment_recall, treatment_F_score))
    elif data_year == 2018:
        print("独立症状:\t\t%.4f\t%.4f\t%.4f" % (signs_precision, signs_recall, signs_F_score))
        print("解刨部位:\t\t%.4f\t%.4f\t%.4f" % (body_precision, body_recall, body_F_score))
        print("手术:\t\t%.4f\t%.4f\t%.4f" % (disease_precision, disease_recall, disease_F_score))
        print("症状描述:\t\t%.4f\t%.4f\t%.4f" % (check_precision, check_recall, check_F_score))
        print("药物:\t\t%.4f\t%.4f\t%.4f" % (treatment_precision, treatment_recall, treatment_F_score))
    elif data_year == 2019 or data_year == 2020:
        print("实验室检验:\t%.4f\t%.4f\t%.4f" % (signs_precision, signs_recall, signs_F_score))
        print("解剖部位:\t\t%.4f\t%.4f\t%.4f" % (body_precision, body_recall, body_F_score))
        print("疾病和诊断:\t%.4f\t%.4f\t%.4f" % (disease_precision, disease_recall, disease_F_score))
        print("影像检验:\t\t%.4f\t%.4f\t%.4f" % (check_precision, check_recall, check_F_score))
        print("手术:\t\t%.4f\t%.4f\t%.4f" % (treatment_precision, treatment_recall, treatment_F_score))
        print("药物:\t\t%.4f\t%.4f\t%.4f" % (medical_precision, medical_recall, medical_F_score))
    print("all:\t\t%.4f\t%.4f\t%.4f" % (Precision, Recall, F_score))

    if data_year == 2017 or data_year == 2018:
        print("signs_correct:", signs_correct)
        print("signs_gold:", signs_gold)
        print("signs_predict:", signs_predict)
        print("body_correct:", body_correct)
        print("body_gold:", body_gold)
        print("body_predict:", body_predict)
        print("disease_correct:", disease_correct)
        print("disease_gold:", disease_gold)
        print("disease_predict:", disease_predict)
        print("check_correct:", check_correct)
        print("check_gold:", check_gold)
        print("check_predict:", check_predict)
        print("treatment_correct:", treatment_correct)
        print("treatment:_gold", treatment_gold)
        print("treatment_predict:", treatment_predict)
    elif data_year == 2019 or data_year == 2020:
        print("signs_correct:", signs_correct)
        print("signs_gold:", signs_gold)
        print("signs_predict:", signs_predict)
        print("body_correct:", body_correct)
        print("body_gold:", body_gold)
        print("body_predict:", body_predict)
        print("disease_correct:", disease_correct)
        print("disease_gold:", disease_gold)
        print("disease_predict:", disease_predict)
        print("check_correct:", check_correct)
        print("check_gold:", check_gold)
        print("check_predict:", check_predict)
        print("treatment_correct:", treatment_correct)
        print("treatment:_gold", treatment_gold)
        print("treatment_predict:", treatment_predict)
        print("medical_correct:", medical_correct)
        print("medical_gold:", medical_gold)
        print("medical_predict:", medical_predict)

    print("tp:", tp)
    print("fp:", fp)
    print("fn:", fn)
    print("gold_num:", gold_num)
    print("predict_num:", predict_num)
    print("correct_num:", correct_num)
    # print("Precision:%.4f" % Precision)
    # print("Recall:%.4f" % Recall)
    # print("F-score:%.4f" % F_score)

    # print("p_pred:", y_pred[y_pred > 1])
    # print("p_pred:", y_pred[y_pred > 1])
    # print("y_true:", y_true[y_true > 1])

    # num_proposed = len(y_pred[y_pred > 1])
    # num_correct = (np.logical_and(y_true == y_pred, y_true > 1)).astype(np.int).sum()
    # num_gold = len(y_true[y_true > 1])

    # try:
    #     precision = float(num_correct) / float(num_proposed)
    # except ZeroDivisionError:
    #     precision = 1.0
    #
    # try:
    #     recall = float(num_correct) / float(num_gold)
    # except ZeroDivisionError:
    #     recall = 1.0
    #
    # try:
    #     f1 = 2 * precision * recall / (precision + recall)
    # except ZeroDivisionError:
    #     if precision * recall == 0:
    #         f1 = 1.0
    #     else:
    #         f1 = 0

    final = f + ".P%.2f_R%.2f_F%.2f" % (Precision, Recall, F_score)
    with open(final, 'w', encoding='utf-8') as fout:
        result = open("temp", "r", encoding='utf-8').read()
        fout.write(f"{result}\n")

        fout.write(f"precision={Precision}\n")
        fout.write(f"recall={Recall}\n")
        fout.write(f"f1={F_score}\n")

    os.remove("temp")

    # print("precision=%.4f" % precision)
    # print("recall=%.4f" % recall)
    # print("f1=%.4f" % f1)
    return Precision, Recall, F_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/01")
    parser.add_argument("--trainset", type=str, default="processed/ccks2019/train.txt")
    parser.add_argument("--validset", type=str, default="processed/ccks2019/test.txt")
    hp = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Bert_BiLSTM_CRF(tag2idx, device).cuda()
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    # optimizer = optim.SGD(model.parameters(), lr=hp.lr)
    # optimizer = optim.Adagrad(model.parameters(), lr=hp.lr)
    # optimizer = optim.Adadelta(model.parameters(), lr=hp.lr)
    # optimizer = optim.RMSprop(model.parameters(), lr=hp.lr)
    # optimizer = optim.Adamax(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss()

    print("batch_size:", hp.batch_size)
    print("learning_rate:", hp.lr)
    print("n_epochs:", hp.n_epochs)
    print("train:", hp.trainset)
    print("valid:", hp.validset)
    print("optimizer: Adam")
    print('Initial model Done')
    # model = nn.DataParallel(model)

    # vocab = load_vocab("./chinese_L-12_H-768_A-12/vocab.txt")
    # train_data = load_data(hp.trainset, max_length=MAX_LEN, label_dic=tag2idx, vocab=vocab)
    # # print(train_data)
    # train_texts = [temp.text for temp in train_data]
    # # print(train_texts)
    # train_ids = torch.LongTensor([temp.input_id for temp in train_data])
    # train_masks = torch.LongTensor([temp.input_mask for temp in train_data])
    # train_tags = torch.LongTensor([temp.label_id for temp in train_data])
    # train_dataset = TensorDataset(train_ids, train_masks, train_tags)
    # train_loader = DataLoader(train_dataset, shuffle=True, batch_size=hp.batch_size)
    #
    # test_data = load_data(hp.validset, max_length=MAX_LEN, label_dic=tag2idx, vocab=vocab)
    # # print(train_data)
    # test_texts = [temp.text for temp in test_data]
    # # print(train_texts)
    # test_ids = torch.LongTensor([temp.input_id for temp in test_data])
    # test_masks = torch.LongTensor([temp.input_mask for temp in test_data])
    # test_tags = torch.LongTensor([temp.label_id for temp in test_data])
    # test_dataset = TensorDataset(test_ids, test_masks, test_tags)
    # test_loader = DataLoader(test_dataset, shuffle=True, batch_size=hp.batch_size)

    # print(train_loader)
    # for i, train_batch in enumerate(train_loader):
    #     sentence, masks, tags = train_batch
    #     print("---")
    #     print("train_text:", len(train_texts[i]))
    #     print("train_batch:", train_batch)
    #     print("sentence:", sentence.size())
    #     print("mask:", masks.size())
    #     print("tags:", tags.size())
    #     break

    train_dataset = NerDataset(hp.trainset)
    eval_dataset = NerDataset(hp.validset)
    print('Load Data Done')

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=pad)

    print('Start Train...')
    for epoch in range(1, hp.n_epochs + 1):  # 每个epoch对dev集进行测试

        train(model, train_iter, optimizer, criterion, device)
        print(f"=========eval at epoch={epoch}=========")
        if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
        fname = os.path.join(hp.logdir, str(epoch))
        precision, recall, f1 = eval(model,  eval_iter, fname, device)

        # torch.save(model.state_dict(), f"{fname}.pt")
        # print(f"weights were saved to {fname}.pt")
