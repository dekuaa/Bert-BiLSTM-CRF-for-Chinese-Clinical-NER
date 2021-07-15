f1 = open('train1.txt', 'w', encoding='utf-8')

VOCAB = ('E-CHECK', 'E-BODY', 'E-SIGNS', 'E-DISEASE', 'E-TREATMENT')

with open("train.txt", "r", encoding='utf-8') as f:

    contents = f.readlines()
    # print(contents)
    for content in contents:
        if content != '\n':
            print(content)
            content = content.split()
            if content[1] in VOCAB:
                # print(content[1])
                content[1] = content[1].replace("E-", "I-")
            if len(content) == 2:
                f1.write(content[0] + " " + content[1] + "\n")
        else:
            f1.write('\n')