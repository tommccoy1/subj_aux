import random
import sys

train_size = 140000
test_size = 10000

prefix = sys.argv[1]

fi = open(prefix + ".raw", "r")


fo_train = open(prefix +  ".train", "w")
fo_test_basic = open(prefix + ".test", "w")
fo_gen = open(prefix + ".gen", "w")

used_dict = {}

count_train = 0
count_basic = 0
count_gen = 0

delList = ["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "AUX1", "AUX2", "AUX3", "AUX4", "AUX5", "AUX6", "AUX7", "AUX8", 
        "VI1", "VI2", "VI3", "VT1", "VT2", "VT3", "VT4", "VT5", "VI", "VT"]
delDict = {}
for item in delList:
    delDict[item] = 1


def questionify(sent):
    sent[-2] = "?"

    if "AUX4" in sent:
        ind = sent.index("AUX4")
    else:
        ind = sent.index("AUX5")

    if sent[ind + 1] == "does":
        sent[ind + 2] = sent[ind + 2][:-1]

    newSent = [sent[ind + 1]] + sent[:ind + 1] + sent[ind + 2:]
    return newSent

def process(sent):
    if sent[-1] == "quest":
        quest = 1
    else:
        quest = 0

    newSent = []
    for word in sent:
        if word not in delDict:
            newSent.append(word)

    newNewSent = []
    prevWord = ""
    for word in newSent:
        if prevWord == "does" and word[-1] == "s":
            newNewSent.append(word[:-1])
        else:
            newNewSent.append(word)
        prevWord = word

    return " ".join(newNewSent)

count_orc = 0
count_src = 0
aux_list = ["can", "may", "will", "might", "must", "would", "could", "should", "mayn't", "won't", "mightn't", "wouldn't", "shouldn't", "shan't", "do", "does", "don't", "doesn't"]
aux_dict = {}
for aux in aux_list:
    aux_dict[aux] = 1

def get_auxes(words):
    aux_set = []
    for word in words:
        if word in aux_dict:
            aux_set.append(word)

    new_aux_set = []
    for aux in aux_set:
        if "do" not in aux:
            new_aux_set.append("aux")
        else:
            new_aux_set.append(aux)

    return new_aux_set


for line in fi:
    if count_train >= train_size and count_basic >= test_size:
        break

    sent = line.strip()
    if sent in used_dict:
        continue

    used_dict[sent] = 1

    words = sent.split()

    if words[3] == "that" or words[3] == "who":
        rel_on_subj = 1
    else:
        rel_on_subj = 0

    choose = random.getrandbits(1)

    quest = random.getrandbits(1)
    if count_train >= train_size and count_basic >= test_size:
        quest = True
        choose = 1
    if quest:
        words.append("quest")
    else:
        words.append("decl")

    if quest:
        result = process(words) + "\t" + process(questionify(words)) + "\n"
    else:
        result = process(words) + "\t" + process(words) + "\n"

    if choose == 0 and count_basic >= test_size:
        choose = 1

    if rel_on_subj and quest:
        if count_gen < test_size:
            words_auxes = get_auxes(words)
            if words_auxes == ["do", "don't"] or words_auxes == ["don't", "do"] or words_auxes == ["does", "doesn't"] or words_auxes == ["doesn't", "does"] or words_auxes == ["aux", "aux"]:
                if words[5] in aux_dict:
                    if count_src <= 6666:
                        fo_gen.write(result)
                        count_gen += 1
                        count_src += 1
                    else:
                        fo_gen.write(result)
                        count_gen += 1
                        count_orc += 1
            #else:
            #    print(words_auxes)
    elif choose == 0 and count_basic < test_size and (not rel_on_subj or not quest):
        if not rel_on_subj or not quest:
            fo_test_basic.write(result)
            count_basic += 1
    elif count_train < train_size:
        fo_train.write(result)
        count_train += 1
    else:
        break

print(count_orc, count_src, count_gen, test_size)

