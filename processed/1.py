f1 = open('train1.txt', 'w', encoding='utf-8')

with open("train.txt", "r", encoding='utf-8') as f:

    contents = f.readlines()
    # print(contents)
    for content in contents:
        if content != '\n':
            print(content)
            if len(content.split()) == 2:
                content = content.replace("	", " ")
                f1.write(content)
        else:
            f1.write('\n')