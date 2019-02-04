
# This code was adapted from the tutorial "Translation with a Sequence to 
# Sequence Network and Attention" by Sean Robertson. It can be found at the
# following URL:
# http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

# You must have PyTorch installed to run this code.
# You can get it from: http://pytorch.org/


# Imports
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import sys
import os

# Functions for tracking time
import time
import math


from seq2seq import EncoderRNN, DecoderRNN, indexesFromSentence, variableFromSentence, variablesFromPair, variablesFromPairList, asMinutes, timeSince, prepareData


random.seed(7)


# Start-of-sentence and end-of-sentence tokens
# The standard seq2seq version only has one EOS. This version has 
# 2 EOS--one signalling that the original sentence should be returned,
# the other signalling it should be reversed.
# I use a 1-hot encoding for all tokens.
SOS_token = 0
EOS_tokenA = 1 # For DECL
EOS_tokenB = 2 # For QUEST



prefix = sys.argv[1] # This means we're using the language with agreement
directory = sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + "_" + sys.argv[5] + "_" + sys.argv[6]

counter = 0
dir_made = 0
# Reading the training data
trainingFile = prefix + '.train'
testFile = prefix + '.test'
genFile = prefix + '.gen'

# This affects how training proceeds. I've run it with the default value
# of 0.5 but am trying it with 0.0 now.
teacher_forcing_ratio = 0.5
batch_size = 5
MAX_LENGTH = 20

use_cuda = torch.cuda.is_available()

# Using the encodings for each token
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {u"decl": 1, u"quest": 2}
        self.word2count = {u"decl": 0, u"quest": 0}
        self.index2word = {0: "SOS", 1: "decl", 2: "quest"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    #s = re.sub(r"([.!?])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Add training
    # Read the file and split into lines
    lines = open(trainingFile, encoding='utf-8').\
        read().strip().split('\n') 

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

posDict = {}
posDict["SOS"] = ""
posDict["sos"] = ""
posDict["DECL"] = ""
posDict["decl"] = ""
posDict["QUEST"] = ""
posDict["quest"] = ""
posDict["the"] = "D"
posDict["some"] = "D"
posDict["my"] = "D"
posDict["your"] = "D"
posDict["our"] = "D"
posDict["her"] = "D"
posDict["newt"] = "N"
posDict["newts"] = "N"
posDict["orangutan"] = "N"
posDict["orangutans"] = "N"
posDict["peacock"] = "N"
posDict["peacocks"] = "N"
posDict["quail"] = "N"
posDict["quails"] = "N"
posDict["raven"] = "N"
posDict["ravens"] = "N"
posDict["salamander"] = "N"
posDict["salamanders"] = "N"
posDict["tyrannosaurus"] = "N"
posDict["tyrannosauruses"] = "N"
posDict["unicorn"] = "N"
posDict["unicorns"] = "N"
posDict["vulture"] = "N"
posDict["vultures"] = "N"
posDict["walrus"] = "N"
posDict["walruses"] = "N"
posDict["xylophone"] = "N"
posDict["xylophones"] = "N"
posDict["yak"] = "N"
posDict["yaks"] = "N"
posDict["zebra"] = "N"
posDict["zebras"] = "N"
posDict["giggle"] = "V"
posDict["smile"] = "V"
posDict["sleep"] = "V"
posDict["swim"] = "V"
posDict["wait"] = "V"
posDict["move"] = "V"
posDict["change"] = "V"
posDict["read"] = "V"
posDict["eat"] = "V"
posDict["entertain"] = "V"
posDict["amuse"] = "V"
posDict["high_five"] = "V"
posDict["applaud"] = "V"
posDict["confuse"] = "V"
posDict["admire"] = "V"
posDict["accept"] = "V"
posDict["remember"] = "V"
posDict["comfort"] = "V"
posDict["can"] = "A"
posDict["will"] = "A"
posDict["could"] = "A"
posDict["would"] = "A"
posDict["do"] = "A"
posDict["don't"] = "A"
posDict["does"] = "A"
posDict["doesn't"] = "A"
posDict["around"] = "P"
posDict["near"] = "P"
posDict["with"] = "P"
posDict["upon"] = "P"
posDict["by"] = "P"
posDict["behind"] = "P"
posDict["above"] = "P"
posDict["below"] = "P"
posDict["who"] = "R"
posDict["that"] = "R"
posDict["."] = "T" # for puncTuation
posDict["?"] = "T" # while this one is for punctuaTion
posDict["+Q"] = "K"
posDict["+q"] = "K"
posDict["-Q"] = "K"
posDict["-q"] = "K"
posDict["t"] = "C"


def sentToPos(sent):
        poses = []
        for word in sent.split():
            poses.append(posDict[word])
        return " ".join(poses)


input_lang, output_lang, pairs, batches, dict_to_check, MAX_LENGTH = prepareData('eng', 'fra', batch_size, False)
print(random.choice(pairs))

print("Choices from batches")
print(batches[0])
print(batches[1])

recurrent_unit = sys.argv[3] # Could be "SRN" or "LSTM" instead
attention = sys.argv[4]# Could be "n" instead

if attention == "0":
        attention = 0
elif attention == "1":
        attention = 1
elif attention == "2":
	attention = 2
else:
        print("Please specify 'y' for attention or 'n' for no attention.")


MAX_EXAMPLE = 10000
def evaluate(encoder, decoder, batch, max_length=MAX_LENGTH):
    input_pair = variablesFromPairList(batch)
    #print(batch, "batch")
    #print(input_pair, "input_pair")

    input_variable = input_pair[0]
    target_variable = input_pair[1]

    encoder_hidden = encoder.initHidden(recurrent_unit)

    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden(recurrent_unit)

    encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    target_length = target_variable.size()[0]

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output

    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoder_outputs = []

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output, encoder_outputs, input_variable, attn=attention)
        topv, topi = decoder_output.data.topk(1)
        decoder_input = Variable(topi.view(-1))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_outputs.append(topi) 

        if EOS_tokenA in topi[0] or EOS_tokenB in topi[0]:
                break

    return decoder_outputs, None

auxes = ["can", "could", "will", "would", "do", "does", "don't", "doesn't"]
def crain(sentence, output):
        index1 = -1
        index2 = -1

        words = sentence.replace("?", ".").replace("decl", "quest").replace("DECL", "quest").replace("QUEST", "quest").split()

        for ind, word in enumerate(words):
                if word in auxes:
                        if index1 == -1:
                                index1 = ind
                        elif index2 == -1:
                                index2 = ind

        aux1 = words[index1]
        aux2 = words[index2]

        d1 = " ".join(words[:index1] + words[index1 + 1:])
        d2 = " ".join(words[:index2] + words[index2 + 1:])
        dn = " ".join(words)

        output = output.replace("?", ".").replace("decl", "quest").replace("DECL", "quest").replace("QUEST", "quest")

        #print(output)
        if output == aux1 + " " + d1:
                return "d1p1"
        if output == aux2 + " " + d1:
                return "d1p2"
        if output == aux1 + " " + d2:
                return "d2p1"
        if output == aux2 + " " + d2:
                return "d2p2"
        if output == aux1 + " " + dn:
                return "dnp1"
        if output == aux2 + " " + dn:
                return "dnp2"
        if output.split()[0] in auxes:
                if output.split()[1:] == d1:
                        return "d1po"
                if output.split()[1:] == d2:
                        return "d2po"
                if output.split()[1:] == dn:
                        return "dnpo"

        return "other"
# Show the output for a few randomly selected sentences
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        batch = random.choice(test_batches)
        #print('>', pair[0])
        #print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, batch)

        for index in range(batch_size):
                this_sent = []
                for output_word in output_words:
                        this_sent.append(output_lang.index2word[output_word[index].item()])
                        if output_lang.index2word[output_word[index].item()] == "decl" or output_lang.index2word[output_word[index].item()] == "quest":
                            break
                if index == 0:
                        to_annotate = batch[index][0].split()
                        print(batch[index][0])
                        print(batch[index][1])
                        print(" ".join(this_sent))
                        print(" ")

    for i in range(n):
        batch = random.choice(gen_batches)
        #print('>', pair[0])
        #print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, batch)

        for index in range(batch_size):
                this_sent = []
                for output_word in output_words:
                        this_sent.append(output_lang.index2word[output_word[index].item()])
                if index == 0:
                        to_annotate = batch[index][0].split()
                        print(batch[index][0])
                        print(batch[index][1])
                        print(" ".join(this_sent))
                        print(crain(batch[index][0], " ".join(this_sent)))
                        print(" ")



# Returns the second auxiliary in a sentence
# Assumes that the sentence has (at least) 2 auxiliaries in it
def second_aux(sent):
    seen_aux = 0
    
    for word in sent:
        if seen_aux:
            if word in ["do", "does", "don't", "doesn't"]:
                return word
        else:
            if word in ["do", "does", "don't", "doesn't"]:
                seen_aux = 1



# Where the actual running of the code happens
hidden_size = int(sys.argv[6]) # Default 128
encoder1 = EncoderRNN(input_lang.n_words, hidden_size, recurrent_unit)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words, recurrent_unit, attn=attention, n_layers=1, dropout_p=0.1)

if use_cuda:
    encoder1 = encoder1.cuda()
    decoder1 = decoder1.cuda()


counter = 0
direcs_to_process = 1

lines = open(testFile, encoding='utf-8').read().strip().split('\n')
test_pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

length_sorted_pairs_dict = {}
for i in range(30):
        length_sorted_pairs_dict[i] = []

for pair in test_pairs:
        length = len(pair[0].strip().split("\t")[0].split())
        length_sorted_pairs_dict[length].append(pair)

length_sorted_pairs_list = []
for i in range(30):
        possibilities = length_sorted_pairs_dict[i]
        random.shuffle(possibilities)

        used_up = 0
        this_set = []
        for j in range(len(possibilities)):
                this_set.append(possibilities[j])
                if len(this_set) == batch_size:
                        length_sorted_pairs_list.append(this_set)
                        this_set = []

test_batches = length_sorted_pairs_list
random.shuffle(test_batches)

print("batch examples")
print(test_batches[0])
print(" ")
print(test_batches[1])


lines = open(genFile, encoding='utf-8').read().strip().split('\n')
gen_pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

length_sorted_pairs_dict = {}
for i in range(30):
	length_sorted_pairs_dict[i] = []

for pair in gen_pairs:
        length = len(pair[0].strip().split("\t")[0].split())
        length_sorted_pairs_dict[length].append(pair)

length_sorted_pairs_list = []
for i in range(30):
        possibilities = length_sorted_pairs_dict[i]
        random.shuffle(possibilities)

        used_up = 0
        this_set = []
        for j in range(len(possibilities)):
                this_set.append(possibilities[j])
                if len(this_set) == batch_size:
                        length_sorted_pairs_list.append(this_set)
                        this_set = []

gen_batches = length_sorted_pairs_list[:]
random.shuffle(gen_batches)

allTotalTest = 0
allTestCorrect = 0
allTestCorrectPos = 0
allTotalGen = 0
allGenCorrect = 0
allGenFullsent = 0
allGenFullsentPos = 0
allGenFirstAux = 0
allGenOtherAux = 0
allGenOtherWord = 0

d1p1 = 0
d1p2 = 0
d1po = 0
d2p1 = 0
d2p2 = 0
d2po = 0
dnp1 = 0
dnp2 = 0
dnpo = 0
other_crain = 0

test_full_sent = []
test_full_sent_pos = []
gen_full_sent = []
gen_full_sent_pos = []
gen_first_word = []
gen_first_word_first_aux = []
gen_first_word_other_aux = []
gen_first_word_other_word = []

while direcs_to_process:
        if not os.path.exists(directory + "_" +  str(counter)):
                direcs_to_process = 0
        else:
                directory_now = directory + "_" + str(counter)
                counter += 1
		
                dec_list = sorted(os.listdir(directory_now))
                dec = sorted(dec_list[:int(len(dec_list)/2)], key=lambda x:float(".".join(x.split(".")[2:4])))[0]
                print("This directory:", dec)
                enc = dec.replace("decoder", "encoder")

                encoder1.load_state_dict(torch.load(directory_now + "/" + enc))
                decoder1.load_state_dict(torch.load(directory_now + "/" + dec))


                evaluateRandomly(encoder1, decoder1)
                right = 0
                rightpos = 0
                total = 0

                test = open(testFile, "r")
                test_set = test.readlines()

                for batch in test_batches[200:400]:
                        pred_words, att = evaluate(encoder1, decoder1, batch)
                        #print(pred_words)
                        for index in range(batch_size):
                                this_sent = []
                                for output_word in pred_words:
                                        #print(output_word, "output_word")
                                        #print(output_word[index], "output word at index")
                                        this_sent.append(output_lang.index2word[output_word[index].item()])
                                if "decl" in this_sent:
                                        this_sent = this_sent[:this_sent.index("decl") + 1]
                                if "quest" in this_sent:
                                        this_sent = this_sent[:this_sent.index("quest") + 1]
                                this_sent_final = " ".join(this_sent)
                                total += 1
                                #print(this_sent_final, batch[index][1])
                                if this_sent_final == batch[index][1]:
                                        right += 1
                                if sentToPos(this_sent_final) == sentToPos(batch[index][1]):
                                        rightpos += 1

			
                print("Test number correct:", right)
                print("Test total:", total)

                allTestCorrect += right
                allTotalTest += total
                allTestCorrectPos += rightpos
                test_full_sent.append(right * 1.0 / total)
                test_full_sent_pos.append(rightpos * 1.0 / total)

                # Counts for how many sentences in the generalization set was the correct auxiliary predicted
                right = 0
                first_aux = 0
                other_aux = 0
                other_word = 0
                total = 0
                other = 0
                full_right = 0
                full_right_pos = 0

                for batch in gen_batches[:400]:
                        output_words, att = evaluate(encoder1, decoder1, batch)
                        for index in range(batch_size):

                                this_sent = []
                                for output_word in output_words:
                                        this_sent.append(output_lang.index2word[output_word[index].item()])
                          
                                if "decl" in this_sent:
                                        this_sent = this_sent[:this_sent.index("decl") + 1]
                                if "quest" in this_sent:
                                        this_sent = this_sent[:this_sent.index("quest") + 1]
                           
                                this_sent_final = " ".join(this_sent)
                                correct_words = batch[index][1].split()
                                count_can = 0
                                count_will = 0
                                count_could = 0
                                count_would = 0

                                #print(correct_words, "corr_words")
                                #print(this_sent_final, "this_sent_final")

                                count_aux = 0
                                for word in correct_words:
                                        if count_aux == 2:
                                                break
                                        if word == "do" or word == "can":
                                                count_can += 1
                                                count_aux += 1
                                        elif word == "does" or word == "will":
                                                count_will += 1
                                                count_aux += 1
                                        elif word == "don't" or word == "could":
                                                count_could += 1
                                                count_aux += 1
                                        elif word == "doesn't" or word == "would":
                                                count_would += 1
                                                count_aux += 1
                                if count_can + count_will + count_could + count_would == 2:
                                        if count_can != 2 and count_will != 2 and count_could != 2 and count_would != 2:
                                                if (count_can + count_could == 2) or (count_will + count_would == 2):
                                                        total += 1
                                                        #print(this_sent_final.split()[0])
                                                        #print(batch[index][1].split()[0])
                                                        if this_sent_final.split()[0] == batch[index][1].split()[0]:
                                                                right += 1
                                                        elif this_sent_final.split()[0] in batch[index][1].split() and this_sent_final.split()[0] in auxes: 
                                                                first_aux += 1
                                                        elif this_sent_final.split()[0] in auxes:
                                                                other_aux += 1
                                                        else:
                                                                other_word += 1

                                                        if this_sent_final == batch[index][1]:
                                                                full_right += 1
                                                        if sentToPos(this_sent_final) == sentToPos(batch[index][1]):
                                                                full_right_pos += 1

                                                        crain_class = crain(batch[index][0], this_sent_final)
                                                        #print("this sent final:")
                                                        #print(this_sent_final)
                                                        #print(batch[index][1])
                                                        #print(crain_class + "\n")
                                                        if crain_class == "d1p1":
                                                                d1p1 += 1
                                                        elif crain_class == "d1p2":
                                                                d1p2 += 1
                                                        elif crain_class == "d1po":
                                                                d1po += 1
                                                        elif crain_class == "d2p1":
                                                                d2p1 += 1
                                                        elif crain_class == "d2p2":
                                                                d2p2 += 1
                                                        elif crain_class == "d2po":
                                                                d2po += 1
                                                        elif crain_class == "dnp1":
                                                                dnp1 += 1
                                                        elif crain_class == "dnp2":
                                                                dnp2 += 1
                                                        elif crain_class == "dnpo":
                                                                dnpo += 1
                                                        else:
                                                                other_crain += 1
                                                                #print("OTHER", other)




                                                        #print(this_sent_final, batch[index][1].split())
                print("Number of sentences with the correct prediction:", right)
                print("Number of sentences fully correct:", full_right)
                print("Total number of sentences", total)
		
                allTotalGen += total
                allGenCorrect += right
                allGenFirstAux += first_aux
                allGenOtherAux += other_aux
                allGenOtherWord += other_word
                allGenFullsent += full_right
                allGenFullsentPos += full_right_pos
                
                gen_full_sent.append(full_right * 1.0 / total)
                gen_full_sent_pos.append(full_right_pos * 1.0 / total)
                gen_first_word.append(right * 1.0 / total)
                gen_first_word_first_aux.append(first_aux * 1.0/total)
                gen_first_word_other_aux.append(other_aux * 1.0 / total)
                gen_first_word_other_word.append(other_word * 1.0 / total)

print("Overall test correct:", allTestCorrect)
print("Overall test total:", allTotalTest)
print("Overall test accuracy:", allTestCorrect * 1.0 / allTotalTest)
print("Test accuracy list:")
print(", ".join([str(x) for x in test_full_sent]))
print(" ")
print("Overall test correct POS:", allTestCorrectPos)
print("Overall test total:", allTotalTest)
print("Overall test accuracy:", allTestCorrectPos * 1.0 / allTotalTest)
print("Test accuracy list:")
print(", ".join([str(x) for x in test_full_sent_pos]))
print(" ")

print("Overall gen first word correct aux:", allGenCorrect)
print("Overall gen total:", allTotalGen)
print("Overall gen accuracy", allGenCorrect * 1.0 / allTotalGen)
print("Gen first word list:")
print(", ".join([str(x) for x in gen_first_word]))
print(" ")
print("Overall gen first word first aux:", allGenFirstAux)
print("Overall gen total:", allTotalGen)
print("Overall gen accuracy", allGenFirstAux * 1.0 / allTotalGen)
print("Gen first word list first:")
print(", ".join([str(x) for x in gen_first_word_first_aux]))
print(" ")
print("Overall gen first word other aux:", allGenOtherAux)
print("Overall gen total:", allTotalGen)
print("Overall gen accuracy", allGenOtherAux * 1.0 / allTotalGen)
print("Gen first word list:")
print(", ".join([str(x) for x in gen_first_word_other_aux]))
print(" ")
print("Overall gen first word other word:", allGenOtherWord)
print("Overall gen total:", allTotalGen)
print("Overall gen accuracy", allGenOtherWord * 1.0 / allTotalGen)
print("Gen first word list:")
print(", ".join([str(x) for x in gen_first_word_other_word]))
print(" ")


print("Overall gen full sentence correct:", allGenFullsent)
print("Overall gen total:", allTotalGen)
print("Overall gen accuracy", allGenFullsent * 1.0 / allTotalGen)
print("Gen full sentence list:")
print(", ".join([str(x) for x in gen_full_sent]))
print(" ")
print("Overall gen full sentence POS correct:", allGenFullsentPos)
print("Overall gen total:", allTotalGen)
print("Overall gen accuracy", allGenFullsentPos * 1.0 / allTotalGen)
print("Gen full sentence list:")
print(", ".join([str(x) for x in gen_full_sent_pos]))
print(" ")



print("d1p1", d1p1 * 1.0 / allTotalGen)
print("d1p2", d1p2 * 1.0 / allTotalGen)
print("d1po", d1po * 1.0 / allTotalGen)
print("d2p1", d2p1 * 1.0 / allTotalGen)
print("d2p2", d2p2 * 1.0 / allTotalGen)
print("d2po", d2po * 1.0 / allTotalGen)
print("dnp1", dnp1 * 1.0 / allTotalGen)
print("dnp2", dnp2 * 1.0 / allTotalGen)
print("dnpo", dnpo * 1.0 / allTotalGen)
print("other", other_crain * 1.0 / allTotalGen)
#print("other", other_crain)
print("d1p2:", d1p2)

