import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch import optim
import os
import json
import numpy as np
import sys

VOCAB_SIZE = 6772
TEST_DIR = os.path.join(sys.argv[1],'testing_data/feat')
LABEL_PATH = os.path.join(sys.argv[1],'training_label.json')
ID_PATH = os.path.join(sys.argv[1],'testing_id.txt')
USE_CUDA = torch.cuda.is_available()
BOS_TOKEN = 0
EOS_TOKEN = 1
hidden_size = 256
class Vocab:
	def __init__(self,label_path):
		print("Building Vocab")
		with open(label_path,'r') as f:
			self.label = json.load(f)
		self.vocab2index = {'<BOS>':0, '<EOS>':1}
		self.index2vocab = {0:'<BOS>',1:'<EOS>'}
		self.num_words = 2
		self.build()

	def build(self):
		for l in self.label:
			for line in l["caption"]:
				line = line.replace('.','')
				line = line.replace('!','')
				line = line.replace('(','')
				line = line.replace(')','')
				for w in line.strip().split():
					if w not in self.vocab2index.keys():
						self.vocab2index[w] = self.num_words
						self.index2vocab[self.num_words] = w
						self.num_words += 1			

class Testset(Dataset):
	def __init__(self,data_dir,id_path):
		print("Preparing dataset")
		self.data_dir = data_dir
		self.label = []
		with open(id_path,'r') as f:
                    for line in f:
                        self.label.append(line.strip())
	def __len__(self):
		return len(self.label)
	
	def __getitem__(self,index):
		avi_id = self.label[index]+'.npy'
		data = np.load(os.path.join(self.data_dir,avi_id))
		return data,self.label[index] 


class Encoder(nn.Module):
	def __init__(self, input_size, hidden_size, batch_size):
		super(Encoder, self).__init__()
		self.hidden_size = hidden_size
		self.batch_size = batch_size
		self.lstm1 = nn.LSTM(input_size, hidden_size)
		self.lstm2 = nn.LSTM(2*hidden_size, hidden_size)
		self.padding = torch.zeros(batch_size, hidden_size) 
	def init_hidden(self,batch_size):
		if USE_CUDA:
			return (Variable(torch.zeros(1,batch_size, self.hidden_size)).cuda(),\
				Variable(torch.zeros(1,batch_size, self.hidden_size)).cuda())
		else:
			return (Variable(torch.zeros(1,batch_size, self.hidden_size)),\
				Variable(torch.zeros(1,batch_size, self.hidden_size))) 
	def forward(self, input_feat, hidden1,hidden2):
		self.lstm1.flatten_parameters()
		output1, hidden1 = self.lstm1(input_feat,hidden1) 
		padding = Variable(self.padding.repeat(len(input_feat),1).view(len(input_feat),1,-1))
		padding = padding.cuda() if USE_CUDA else padding
		output2, hidden2 = self.lstm2(torch.cat((padding,output1),2),hidden2)
		#output = self.hidden2out(output.view(output.size()[0],-1))
		#output = self.softmax(output)
		return hidden1, hidden2

class Decoder(nn.Module):
	def __init__(self, input_size, hidden_size, batch_size,lstm1,lstm2):
		super(Decoder, self).__init__()
		self.hidden_size = hidden_size
		self.batch_size = batch_size
		self.lstm1 = lstm1
		self.lstm2 = lstm2
		self.padding = torch.zeros(batch_size, input_size) 
		self.hidden2out = nn.Linear(hidden_size,VOCAB_SIZE)
		self.vocab2emb = nn.Linear(VOCAB_SIZE,hidden_size)
		#self.dropout = nn.Dropout(0.4)
		self.softmax = nn.LogSoftmax()
	def forward(self, hidden1,hidden2,target_length,eva = False):
		padding = Variable(self.padding.repeat(target_length,1).view(target_length,self.batch_size,-1))
		padding = padding.cuda() if USE_CUDA else padding
		output1, hidden1 = self.lstm1(padding,hidden1) 
		bos_onehot = Variable(torch.FloatTensor([1]+[0 for _ in range(VOCAB_SIZE-1)]))
		bos_onehot = bos_onehot.cuda() if USE_CUDA else bos_onehot
		output2_vocab_emb = self.vocab2emb(bos_onehot).view(1,self.batch_size,-1)
		hidden2_t = hidden2
		output2 = Variable(torch.zeros(target_length,self.batch_size,VOCAB_SIZE))
		output2 = output2.cuda() if USE_CUDA else output2
		if eva == False:
			for i in range(target_length):
				output2_t, hidden2_t = self.lstm2(torch.cat((output2_vocab_emb,output1[i].view(1,self.batch_size,-1)),2),hidden2_t)
				output2_t = self.softmax(self.hidden2out(output2_t.view(output2_t.size()[0],-1)))
				output2[i] = output2_t
				output2_vocab_emb = self.vocab2emb(output2_t.view(1,self.batch_size,-1))
		return output2

if USE_CUDA:
	print("USE CUDA")
else:
	print("Please use CUDA, because the saved model has torch cuda format.")

V = Vocab(LABEL_PATH)
DS = Testset(TEST_DIR,ID_PATH)
CHECK_PATH = './199.tar'
checkpoint = torch.load(CHECK_PATH)
E = Encoder(4096,hidden_size,1)
E.load_state_dict(checkpoint['encoder'])
h1 = E.init_hidden(1)
h2 = E.init_hidden(1)
D = Decoder(4096,hidden_size,1,E.lstm1,E.lstm2)
D.load_state_dict(checkpoint['decoder'])
MAX_LENGTH = 30
if USE_CUDA:
	E = E.cuda()
	D = D.cuda()

def output_sen(index_list,V):
	output = ""
	for i in index_list:
		if i == 1:
			#output += "."
			break
		elif i == 0:
			continue
		else:
			output += V.index2vocab[i]+" "
	return output	

if sys.argv[3] == 'special':
	special_dict = {'klteYv1Uv9A_27_33.avi':'','5YJaS2Eswg0_22_26.avi':'','UbmZAe5u5FI_132_141.avi':'','JntMAcTlOF0_50_70.avi':'','tJHUH9tpqPg_113_118.avi':''}

for data in DS:
	ID = data[1]
	if sys.argv[3] == 'special':
		if ID not in special_dict.keys():
			continue
		else:
			feat = data[0]
			feat = Variable(torch.from_numpy(feat).view(feat.shape[0],1,-1).float())
			feat = feat.cuda() if USE_CUDA else feat
			hidden1, hidden2 = E(feat,h1,h2)
			output = D(hidden1,hidden2,MAX_LENGTH)
			output = output.view(-1,VOCAB_SIZE)
			_,index = torch.max(output,1)
			special_dict[ID] = output_sen(index.data.tolist(),V)
with open(sys.argv[2],'w') as f:
	for k,v in special_dict.items():
		f.write('{},{}\n'.format(k,v))
