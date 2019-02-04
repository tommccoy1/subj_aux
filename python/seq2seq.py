
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


if __name__ == "__main__":
        random.seed(int(sys.argv[7]))
        wait_time = random.randint(0, 99)
        time.sleep(wait_time)

        counter = 0
        dir_made = 0

        while not dir_made:
                if not os.path.exists(directory + "_" +  str(counter)):
                        directory = directory + "_" + str(counter)
                        os.mkdir(directory)
                        dir_made = 1

                else:
                        counter += 1

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
        self.word2index = {u"past": 1, u"present": 2}
        self.word2count = {u"past": 0, u"present": 0}
        self.index2word = {0: "SOS", 1: "past", 2: "present"}
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

# Preparing data--also not really applicable here because we've already
# restricted the lengths of our training data. (This does lowercase the
# input and remove punctuation, but those things also don't matter here)
def prepareData(lang1, lang2, batch_size, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    length_sorted_pairs_dict = {}
    for i in range(30):
        length_sorted_pairs_dict[i] = []  
    MAX_LENGTH = 0

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[0])
        input_lang.addSentence(pair[1])
        output_lang.addSentence(pair[1])

        length = len(pair[0].strip().split("\t")[0].split())
        if length not in length_sorted_pairs_dict:
                length_sorted_pairs_dict[length] = []
        length_sorted_pairs_dict[length].append(pair)

        if length > MAX_LENGTH:
                MAX_LENGTH = length
    for test_line in open(testFile, encoding='utf-8'):
        parts = test_line.strip().split("\t")
        input_lang.addSentence(parts[0])
        output_lang.addSentence(parts[0])
        input_lang.addSentence(parts[1])
        output_lang.addSentence(parts[1])



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
    random.shuffle(length_sorted_pairs_list)

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs, length_sorted_pairs_list, length_sorted_pairs_dict, MAX_LENGTH



input_lang, output_lang, pairs, batches, dict_to_check, MAX_LENGTH = prepareData('eng', 'fra', batch_size, False)
print(random.choice(pairs))

# Note: nn.Linear contains a bias term within it, which 
# is why no bias term appears in the class below
class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.wi = nn.Linear(hidden_size + input_size, hidden_size)
        self.wf = nn.Linear(hidden_size + input_size, hidden_size)
        self.wg = nn.Linear(hidden_size + input_size, hidden_size)
        self.wo = nn.Linear(hidden_size + input_size, hidden_size)


    def forward(self, input, hidden):
        hx, cx = hidden
        input_plus_hidden = torch.cat((input, hx), 2)
        i_t = F.sigmoid(self.wi(input_plus_hidden))
        f_t = F.sigmoid(self.wf(input_plus_hidden))
        g_t = F.tanh(self.wg(input_plus_hidden))
        o_t = F.sigmoid(self.wo(input_plus_hidden))

        cx = f_t * cx + i_t * g_t
        hx = o_t * F.tanh(cx)

        return hx, (hx, cx)

class CumMax(nn.Module):
	def __init__(self):
		super(CumMax, self).__init__()

	def forward(self, input):
		#print(nn.Softmax(dim=0)(input))
		#print(nn.Softmax(dim=1)(input))
		#print(nn.Softmax(dim=2)(input))
		#print(torch.cumsum(nn.Softmax()(input), 0))
		#print(torch.cumsum(nn.Softmax()(input), 1))
		#print(torch.cumsum(nn.Softmax(dim=2)(input), 2))

		return torch.cumsum(nn.Softmax(dim=2)(input), 2)


class ONLSTM(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(ONLSTM, self).__init__()

		self.hidden_size = hidden_size
		self.input_size = input_size

		self.wi = nn.Linear(hidden_size + input_size, hidden_size)
		self.wf = nn.Linear(hidden_size + input_size, hidden_size)
		self.wg = nn.Linear(hidden_size + input_size, hidden_size)
		self.wo = nn.Linear(hidden_size + input_size, hidden_size)
		self.wftilde = nn.Linear(hidden_size + input_size, hidden_size)
		self.witilde = nn.Linear(hidden_size + input_size, hidden_size)




	def forward(self, input, hidden):
		hx, cx = hidden
		input_plus_hidden = torch.cat((input, hx), 2)

		f_t = F.sigmoid(self.wf(input_plus_hidden))
		i_t = F.sigmoid(self.wi(input_plus_hidden))
		o_t = F.sigmoid(self.wo(input_plus_hidden))
		c_hat_t = F.tanh(self.wg(input_plus_hidden))
		
		f_tilde_t = CumMax()(self.wftilde(input_plus_hidden))
		i_tilde_t = 1 - CumMax()(self.witilde(input_plus_hidden))

		omega_t = f_tilde_t * i_tilde_t
		f_hat_t = f_t * omega_t + (f_tilde_t - omega_t)
		i_hat_t = i_t * omega_t + (i_tilde_t - omega_t)

		cx = f_hat_t * cx + i_hat_t * c_hat_t
		hx = o_t * F.tanh(cx)

		return hx, (hx, cx)



# Note: nn.Linear contains a bias term within it, which 
# is why no bias term appears in the class below
class LSTMSqueeze(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMSqueeze, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.wi = nn.Linear(hidden_size + input_size, hidden_size)
        self.wf = nn.Linear(hidden_size + input_size, hidden_size)
        self.wg = nn.Linear(hidden_size + input_size, hidden_size)
        self.wo = nn.Linear(hidden_size + input_size, hidden_size)


    def forward(self, input, hidden):
        hx, cx = hidden
        input_plus_hidden = torch.cat((input, hx), 2)
        i_t = F.sigmoid(self.wi(input_plus_hidden))
        f_t = F.sigmoid(self.wf(input_plus_hidden))
        g_t = F.tanh(self.wg(input_plus_hidden))
        o_t = F.sigmoid(self.wo(input_plus_hidden))

	# Sigmoid as a method to squeeze it
        #cx = F.sigmoid(f_t * cx + i_t * g_t)

	# Halving as a method to squeeze it
        cx = (f_t * cx + i_t * g_t)/2
        hx = o_t * F.tanh(cx)

        return hx, (hx, cx)






# Class for the encoder RNN
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, recurrent_unit, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)

        if recurrent_unit == "SRN":
                self.rnn = nn.RNN(hidden_size, hidden_size)
        elif recurrent_unit == "GRU":
                self.rnn = nn.GRU(hidden_size, hidden_size)
        elif recurrent_unit == "LSTM":
                self.rnn = nn.LSTM(hidden_size, hidden_size)
        elif recurrent_unit == "MyLSTM":
                self.rnn = MyLSTM(hidden_size, hidden_size)
        elif recurrent_unit == "LSTMSqueeze":
                self.rnn = LSTMSqueeze(hidden_size, hidden_size)
        elif recurrent_unit == "ONLSTM":
                self.rnn = ONLSTM(hidden_size, hidden_size)
        else:
                print("Invalid recurrent unit type")

    # For succesively generating each new output and hidden layer
    def forward(self, input, hidden):
        output = self.embedding(input).unsqueeze(0)#.view(1, 1, -1)
        for i in range(self.n_layers):
            output, hidden = self.rnn(output, hidden)
        return output, hidden


    # Creates the initial hidden state
    def initHidden(self, recurrent_unit):
        if recurrent_unit == "SRN" or recurrent_unit == "GRU":
                result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        elif recurrent_unit == "LSTM" or recurrent_unit == "MyLSTM" or recurrent_unit == "LSTMSqueeze" or recurrent_unit == "ONLSTM":
                result = (Variable(torch.zeros(1, batch_size, self.hidden_size)), Variable(torch.zeros(1, batch_size, self.hidden_size)))   
        else:
                print("Invalid recurrent unit type")

        if use_cuda:
                if recurrent_unit == "LSTM" or recurrent_unit == "MyLSTM" or recurrent_unit == "LSTMSqueeze" or recurrent_unit == "ONLSTM":
                        return (result[0].cuda(), result[1].cuda())
                else:
                        return result.cuda()
        else:
                return result

# Class for the basic decoder RNN, without attention
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, recurrent_unit, attn=False, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        if recurrent_unit == "SRN":
                self.rnn = nn.RNN(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "GRU":
                self.rnn = nn.GRU(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "LSTM":
                self.rnn = nn.LSTM(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "MyLSTM":
                self.rnn = MyLSTM(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "LSTMSqueeze":
                self.rnn = LSTMSqueeze(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "ONLSTM":
                self.rnn = ONLSTM(self.hidden_size, self.hidden_size)
        else:
                print("Invalid recurrent unit type")

        self.out = nn.Linear(self.hidden_size, self.output_size)
        
        self.recurrent_unit = recurrent_unit
        
        if attn == 1:
                # Attention vector
                self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

                # Context vector made by combining the attentions
                self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
    
        if attn == 2: # for the other type of attention
                self.v = nn.Parameter(torch.FloatTensor(hidden_size), requires_grad=True)
                nn.init.uniform(self.v, -1, 1) # maybe need cuda
                self.attn_layer = nn.Linear(self.hidden_size * 3, self.hidden_size)
                self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

    # For successively generating each new output and hidden layer
    def forward(self, input, hidden, encoder_output, encoder_outputs, input_variable, attn=False):
        output = self.embedding(input).unsqueeze(0)#.view(1, 1, -1)
        output = self.dropout(output)

        attn_weights = None
        if attn == 1:
		#print(output)
		#print("is the output")
		#print(" ")
		#print(hidden)
		#print("is the hidden")
		#print(" ")
                if self.recurrent_unit == "LSTM" or self.recurrent_unit == "MyLSTM" or self.recurrent_unit == "LSTMSqueeze" or self.recurrent_unit == "ONLSTM":
                    attn_weights = F.softmax(self.attn(torch.cat((output[0], hidden[0][0]), 1)))
                else:
                    attn_weights = F.softmax(self.attn(torch.cat((output[0], hidden[0]), 1)))
		
		#print(attn_weights.unsqueeze(1))
		#print(encoder_outputs.transpose(0,1))
                #attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
                attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.transpose(0,1))
		#print(attn_applied)
                attn_applied = attn_applied.transpose(0,1)

		#print(output)
		#print(attn_applied)

                output = torch.cat((output[0], attn_applied[0]), 1)
		#print(output)
                output = self.attn_combine(output).unsqueeze(0)
		#print(output)


        if attn == 2: # For the other type of attention
		#print("encoder_outputs", encoder_outputs)
		#print("input_variable", input_variable)
                input_length = input_variable.size()[0] # Check if this is the right index
                u_i = Variable(torch.zeros(len(encoder_outputs), batch_size))
		#print("u_i", u_i)

                if use_cuda:
                        u_i = u_i.cuda()
                for i in range(input_length): # can this be done with just matrix operations (i.e. without a for loop)? (probably)
			#print("enc out input", encoder_outputs[i].unsqueeze(0))
			#print("hidden_reshaped", hidden[0].unsqueeze(0))
			#print("output", output)
			#print("output_reshaped", output.unsqueeze(0))

                        if self.recurrent_unit == "LSTM" or self.recurrent_unit == "MyLSTM" or self.recurrent_unit == "LSTMSqueeze" or self.recurrent_unit == "ONLSTM":
                                attn_hidden = F.tanh(self.attn_layer(torch.cat((encoder_outputs[i].unsqueeze(0), hidden[0][0].unsqueeze(0), output), 2)))
                        else:
                                attn_hidden = F.tanh(self.attn_layer(torch.cat((encoder_outputs[i].unsqueeze(0), hidden[0].unsqueeze(0), output), 2))) # the view(-1) is probably bad
			#print("attn_hidden", attn_hidden)
			#print("v", self.v.unsqueeze(1).unsqueeze(0))
                        u_i_j = torch.bmm(attn_hidden, self.v.unsqueeze(1).unsqueeze(0))
			#print("u_i_j", u_i_j)
			#print("u_i_j[0][0][0]", u_i_j[0][0][0])
                        u_i[i] = u_i_j[0].view(-1)

		
                a_i = F.softmax(u_i.transpose(0,1)) # is it correct to be log softmax?
		#print("a_i", a_i)
		#print("a_i_reshaped", a_i.unsqueeze(1))
		#print("enc outputs transpose", encoder_outputs.transpose(0,1))
                attn_applied = torch.bmm(a_i.unsqueeze(1), encoder_outputs.transpose(0,1))
		
		#print("attn_applied", attn_applied)
                attn_applied = attn_applied.transpose(0,1)

		#print("output[0]", output)
                output = torch.cat((output[0], attn_applied[0]), 1)
                output = self.attn_combine(output).unsqueeze(0)
		#print("output_end", output)

        for i in range(self.n_layers):
            #print(output)
            #print(" ")
	    #print(hidden)
            #print(" ")	    
            output = F.relu(output)
            output, hidden = self.rnn(output, hidden)
            
        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights
      
 # Methods for interfacing between words and one-hot encodings
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    #indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)

def variablesFromPairList(pair_list):
	tensorA = None
	tensorB = None
	
	for elt in pair_list:
		this_pair = variablesFromPair(elt)
		if tensorA is None:
			tensorA = this_pair[0]
			tensorB = this_pair[1]
		else:
			#print(tensorB, this_pair[1])
			tensorA = torch.cat((tensorA, this_pair[0]), 1)
			tensorB = torch.cat((tensorB, this_pair[1]), 1)

	return(tensorA, tensorB)







# Training the seq2seq network
def train(training_pair_set, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, recurrent_unit, attention=False, max_length=MAX_LENGTH):
    loss = 0

    for j in [0]:
    #for training_pair in training_pair_set:
        training_pair = training_pair_set
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        encoder_hidden = encoder.initHidden(recurrent_unit)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()


        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
	
        for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output
		#print(encoder_outputs)

        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                        decoder_output, decoder_hidden, decoder_attention = decoder(
                                decoder_input, decoder_hidden, encoder_output, encoder_outputs, input_variable, attn=attention)
                        loss += criterion(decoder_output, target_variable[di])
			#loss += criterion(torch.unsqueeze(decoder_output[0], 0), target_variable[di])
                        decoder_input = target_variable[di]  # Teacher forcing

        else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                        decoder_output, decoder_hidden, decoder_attention = decoder(
                                decoder_input, decoder_hidden, encoder_output, encoder_outputs, input_variable, attn=attention)
                        topv, topi = decoder_output.data.topk(1)
                        #ni = topi[0][0]

                        #decoder_input = Variable(torch.LongTensor([[ni]]))
                        decoder_input = Variable(topi.view(-1))
                        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

                        loss += criterion(decoder_output, target_variable[di])
                        #loss += criterion(torch.unsqueeze(decoder_output[0], 0), target_variable[di])
                        
                        if EOS_tokenA in topi[0] or EOS_tokenB in topi[0]:
                                break
			#if ni == EOS_tokenA or ni == EOS_tokenB:
                        #        break

    loss = loss * batch_size
    if not isinstance(loss, int):
    	loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length
        
 
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))                     
                
                
# Training iterations
def trainIters(encoder, decoder, n_iters, recurrent_unit, attention, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Training with stochastic gradient descent
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    num_pairs = len(pairs)
    training_pairs = [variablesFromPairList(random.choice(batches))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    # Each weight update
    for iter in range(1, n_iters + 1):
        # The iterations we're looping over for this batch
        training_pair_set = training_pairs[(iter - 1)]# * batch_size:iter * batch_size]

        loss = train(training_pair_set, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, recurrent_unit, attention)
        print_loss_total += loss
        plot_loss_total += loss
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        # For saving the weights, if you want to do that
        if iter % 1000 == 0:
                torch.save(encoder.state_dict(), directory + "/" + prefix + ".encoder." + str(print_loss_avg.item()) + "." + str(iter))
                torch.save(decoder.state_dict(), directory + "/" + prefix + ".decoder." + str(print_loss_avg.item()) + "." + str(iter))

if __name__ == "__main__":
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


	# Where the actual running of the code happens
	hidden_size = int(sys.argv[6]) # Default = 128
	encoder1 = EncoderRNN(input_lang.n_words, hidden_size, recurrent_unit)
	decoder1 = DecoderRNN(hidden_size, output_lang.n_words, recurrent_unit, attn=attention, n_layers=1, dropout_p=0.1)

	if use_cuda:
	    encoder1 = encoder1.cuda()
	    decoder1 = decoder1.cuda()
	
	manual_lr = float(sys.argv[5])
	
	torch.manual_seed(int(sys.argv[7]))
	if use_cuda:
        	torch.cuda.manual_seed_all(int(sys.argv[7]))	

	if recurrent_unit == "SRN":
		# Default learning rate: 0.001
		trainIters(encoder1, decoder1, 30000, recurrent_unit, attention, print_every=1000, learning_rate=manual_lr)
	elif attention == 2:
		# Default learning rate: 0.005
		trainIters(encoder1, decoder1, 30000, recurrent_unit, attention, print_every=1000, learning_rate=manual_lr)
	else:
		# Default learning rate: 0.01
		trainIters(encoder1, decoder1, 30000, recurrent_unit, attention, print_every=1000, learning_rate=manual_lr)








