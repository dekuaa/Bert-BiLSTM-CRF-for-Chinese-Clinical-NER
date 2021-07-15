import torch
from pytorch_pretrained_bert import BertTokenizer
from char_featurizer import Featurizer
import char_featurizer
from cnradical import Radical, RunOption

bert_model = 'chinese_L-12_H-768_A-12'
tokenizer = BertTokenizer.from_pretrained(bert_model)

# data = "子"
# featurizer = Featurizer()
# result = featurizer.featurize(data)
# print(result)
# x = []
# for i in range(5):
#     x.append(float(result[i+3][0]))
# x.append(float(result[2][0][0]))
# print(x)

with open("Chinese1.txt", "r", encoding="UTF-8") as f:
    chinese1 = []
    sort1 = []
    answer2 = {}
    contents = f.readlines()
    for content in contents:
        txt = content.split()
        chinese1.append(txt[0])
        sort1.append(txt[1])
        answer2[txt[0]] = txt[1]
f.close()


def process(sentences):
    featurizer = Featurizer()
    radical = Radical(RunOption.Radical)

    character = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ｍ', 'Ｂ', 'ｍ', 'ｇ', 'ｃ', 'Β', 'Ｒ',
                 'Ｕ', 'Ｃ', 'Ｔ', 'Ｈ', 'Ｐ', 'ｒ', 'ａ', 'ｕ', 'ｎ', '[UNK]', 'ｈ', 'Ｅ', 'α',
                 '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '０', '１', '２', '３', '４', '５', '６', '７', '８', '９',
                 'μ', 'μ', 'β', 'π', 'γ', 'σ', 'Ⅲ', 'Ｉ', 'Ⅱ', 'Ⅳ', 'Ⅴ', 'Ⅵ', 'Ⅰ', '⑴', '⑵', '⑶', '㎎', '①', '⑤', '④',
                 '②', '③', 'ⅱ', 'ⅲ', 'ⅴ', 'ⅳ', 'Ⅶ']
    # 无实际含义的定义为符号
    digit = [',', '，', '.', '。', '-', ':', '“', '”', ';', '(', ')', '（', '）', '、', '*', '：', '+', '/', '[', ']', '★',
             '＋', '%', '\\', '=', '＝', '；', '<', '％', '"', '-', '？', '`', '－', '&', '＜', '㎝', '℃', '×', '°', '_',
             '～', '^', '\'', '>', '?', '~', '＞', '‘', '’', '↓', '↑', '㎡', '#', '＂', '【', '】', '±', '—', '．', '／',
             '∕', '《', '》', '［', '］', '✄', '|', '！', '{', '}', '＾']

    # 笔顺
    with open("Chinese.txt", "r", encoding="UTF-8") as f:
        chinese1 = []
        sort1 = []
        answer1 = {}
        contents = f.readlines()
        for content in contents:
            txt = content.split()
            chinese1.append(txt[0])
            sort1.append(txt[1])
            answer1[txt[0]] = txt[1]
    f.close()

    # 偏旁
    with open("Chinese1.txt", "r", encoding="UTF-8") as f:
        chinese1 = []
        sort1 = []
        answer2 = {}
        contents = f.readlines()
        for content in contents:
            txt = content.split()
            chinese1.append(txt[0])
            sort1.append(txt[1])
            answer2[txt[0]] = txt[1]
    f.close()

    max_len = 0
    sen_len = []
    for sentence in sentences:
        words = sentence.split()
        sen_len.append(len(words))
        # for word in words:
        #     if word in tags1:
        #         sen_len[-1] += 3
        #         continue
        #     elif word in tags2:
        #         sen_len[-1] += 2
        #         continue


    for i in range(len(sentences)):
        # print("sen_len:[i]", sen_len[i])
        # print("max_len:", max_len)
        if max_len < sen_len[i]:
            max_len = sen_len[i]
        else:
            continue
    # print("sen_len:", sen_len)
    # print("max_len:", max_len)

    m = []
    for sentence in sentences:
        words = sentence.split()
        # print("words:", len(words))
        # print("words:", words)
        n = []
        # 144 + 5 + 9 = 158
        z = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        # print("z:", len(z))
        for word in words:
            # print("sentence:", sentence)
            # 为汉字
            if word != "[CLS]" and word != "[SEP]" and word != "[PAD]" and word not in character and word not in digit:
                    # and word not in tags1 and word not in tags2:
                a = list(answer1[word])
                # print("a1:", a)
                b = map(eval, a)
                c = []
                for j in b:
                    c.append(j)
                for i in range(144):
                    if i < len(c):
                        c[i] = c[i] + z[i]
                    else:
                        c.append(z[i])
                    i += 1
                result = featurizer.featurize(word)
                # print(result)
                # print("c1:", len(c))
                for i in range(5):
                    c.append(float(result[i + 3][0])/10)
                # print("c2:", len(c))
                # c.append(0.)
                a = [radical.trans_ch(ele) for ele in word]
                a = radical.trans_str(word)
                a = list(answer2[a])
                # print("a2:", a)
                b = map(eval, a)
                for k in b:
                    c.append(k)
                # print("c3:", len(c))
                n.append(c)
            # 为数字或字母
            elif word in character:
                c = [1, 1, 0]
                for i in range(len(z)):
                    if i < len(c):
                        c[i] = c[i] + z[i]
                    else:
                        c.append(z[i])
                    i += 1
                n.append(c)
            # 为[CLS]或[SEP]
            elif word == "[CLS]" or word == "[SEP]" or word == "[PAD]":
                c = [0, 0, 0]
                for i in range(len(z)):
                    if i < len(c):
                        c[i] = c[i] + z[i]
                    else:
                        c.append(z[i])
                    i += 1
                n.append(c)
            # 为标点符号
            elif word in digit:
                c = [1, 1, 1]
                for i in range(len(z)):
                    if i < len(c):
                        c[i] = c[i] + z[i]
                    else:
                        c.append(z[i])
                    i += 1
                n.append(c)
            # elif word in tags1:
            #     c1 = [1, 1, 0]
            #     c2 = [1, 1, 1]
            #     c3 = [1, 1, 0]
            #     c4 = [1, 1, 0]
            #     for i in range(len(z)):
            #         if i < len(c1):
            #             c1[i] = c1[i] + z[i]
            #             c2[i] = c2[i] + z[i]
            #             c3[i] = c3[i] + z[i]
            #             c4[i] = c4[i] + z[i]
            #         else:
            #             c1.append(z[i])
            #             c2.append(z[i])
            #             c3.append(z[i])
            #             c4.append(z[i])
            #         i += 1
            #     n.append(c1)
            #     n.append(c2)
            #     n.append(c3)
            #     n.append(c4)
            # elif word in tags2:
            #     c1 = [1, 1, 0]
            #     c2 = [1, 1, 1]
            #     c3 = [1, 1, 0]
            #     for i in range(len(z)):
            #         if i < len(c1):
            #             c1[i] = c1[i] + z[i]
            #             c2[i] = c2[i] + z[i]
            #             c3[i] = c3[i] + z[i]
            #         else:
            #             c1.append(z[i])
            #             c2.append(z[i])
            #             c3.append(z[i])
            #         i += 1
            #     n.append(c1)
            #     n.append(c2)
            #     n.append(c3)
            # print("c:", len(c))
        # print("n1:", len(n))
        if len(n) < max_len:
            for i in range(max_len - len(n)):
                n.append(z)
        # print("n2:", len(n))

        m.append(n)
    # print(len(m))
    return m
