import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch import optim
import os
import numpy as np
import json
import random
import sys

data_dir = sys.argv[1]

VOCAB_SIZE = 3004 
DATA_DIR = os.path.join(data_dir,'training_data/feat')
TEST_DIR = os.path.join(data_dir,'testing_data/feat')
PEER_DIR = os.path.join(data_dir,'peer_review/feat')
SAVE_DIR = './save'
LABEL_PATH = os.path.join(data_dir,'training_label.json')
ID_PATH = os.path.join(data_dir,'testing_id.txt')
PEER_ID_PATH = os.path.join(data_dir,'peer_review_id.txt')
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
hidden_size = 256 
postfix = 'clip25_lr-3_3000_attn_ss'

class Vocab:
	def __init__(self,label_path):
		print("Building Vocab")
		with open(label_path,'r') as f:
			self.label = json.load(f)
		self.word2index = {'<PAD>':0,'<BOS>':1, '<EOS>':2, '<UNK>':3}
		self.index2word = {0:'<PAD>',1:'<BOS>',2:'<EOS>',3:'<UNK>'}
		self.word2count = {'<PAD>':1,'<BOS>':1,'<EOS>':1,'<UNK>':1}
		self.num_words = 4 
		self.build()
	def build(self):
		for l in self.label:
			for line in l["caption"]:
				for ch in '.!()':
					if ch in line:
						line = line.replace(ch,'')
				for w in line.strip().split():
					if w not in self.word2count.keys():
						self.word2count[w] = 1
					else:
						self.word2count[w] += 1
		sorted_word = [w for (w,c) in sorted(self.word2count.items(), key = lambda x: x[1], reverse = True)]
		for w in sorted_word[:3000]:
			self.word2index[w] = self.num_words
			self.index2word[self.num_words] = w
			self.num_words += 1

def cap2index(caption,V):
	for ch in '.!()':
		if ch in caption:
			caption = caption.replace(ch, '')
	caption_index = [PAD_TOKEN for _ in range(50)]
	for i,word in enumerate(caption.split()):
		if word in V.word2index.keys():
			caption_index[i] = V.word2index[word]
		else:
			caption_index[i] = UNK_TOKEN
	caption_index[len(caption.split())] = EOS_TOKEN
	return caption_index

class TA_Dataset(Dataset):
	def __init__(self,data_dir,label_path):
		print("Preparing dataset")
		self.data_dir = data_dir
		with open(label_path,'r') as f:
			self.label = json.load(f)

	def __len__(self):
		return len(self.label)
	
	def __getitem__(self,index):
		avi_id = self.label[index]["id"]+'.npy'
		data = torch.from_numpy(np.load(os.path.join(self.data_dir,avi_id))).float().cuda()
		caption_index = torch.LongTensor(cap2index(random.choice(self.label[index]["caption"]),V)).cuda()
		return data,caption_index

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
	def __init__(self,input_size,hidden_size,layer=1):
		super(Encoder,self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.layer = layer
		self.lstm = nn.LSTM(input_size,hidden_size,layer,batch_first = True)
		self.attention = nn.Linear(80,80)
	
	def forward(self,data,hidden):
		for i in range(self.layer):
			output,hidden = self.lstm(data,hidden)
		return output,hidden

	def init_hidden(self,batch_size):
		return Variable(torch.zeros(1,batch_size,self.hidden_size).cuda()),Variable(torch.zeros(1,batch_size,self.hidden_size).cuda())

class Decoder(nn.Module):
	def __init__(self,input_size,hidden_size,layer=1):
		super(Decoder,self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.layer = layer
		self.lstm = nn.LSTM(input_size,hidden_size,layer,batch_first = True)
		self.hidden2out = nn.Linear(hidden_size,VOCAB_SIZE)
		self.softmax = nn.LogSoftmax()
		self.embedding = nn.Embedding(VOCAB_SIZE,hidden_size)

	def forward(self,data,hidden):
		for i in range(self.layer):
			output,hidden = self.lstm(data,hidden)
			result = self.softmax(self.hidden2out(output).view(-1,VOCAB_SIZE))
			embed = self.embedding(torch.max(result,1)[1]).view(-1,1,hidden_size)
		return result, output, embed, hidden

	def init_hidden(self,batch_size):
		return Variable(torch.zeros(1,batch_size,self.hidden_size).cuda()),Variable(torch.zeros(1,batch_size,self.hidden_size).cuda())


### Training Stage ###

V = Vocab(LABEL_PATH)
'''
ITER =  200
DS = TA_Dataset(DATA_DIR,LABEL_PATH)
DL = DataLoader(DS,batch_size = 16)
print("Finish building dataloader")
E = Encoder(4096+hidden_size,hidden_size, 1).cuda()
D = Decoder(2*hidden_size, hidden_size, 1).cuda()
criterion = nn.NLLLoss().cuda()
e_optim = optim.Adam(E.parameters(), lr=1e-3)
d_optim = optim.Adam(D.parameters(), lr=1e-3)
SS_RATIO = 0.6

directory = os.path.join(SAVE_DIR,  'S2VT_EnDe', '{}-{}'.format(hidden_size,postfix))
if not os.path.exists(directory):
    os.makedirs(directory)

SAVE = 1	   
for epoch in range(1,ITER+1):
	total_loss = 0
	for data,caption_index in DL:
		e_optim.zero_grad()
		d_optim.zero_grad()
	
		feat = Variable(data.view(data.size()[0], 80, -1))
		target = Variable(caption_index)
		## stage 1 ##

		decoder_padding = Variable(torch.zeros(data.size()[0],80,hidden_size).cuda())
		encoder_hidden = E.init_hidden(data.size()[0])
		decoder_hidden = D.init_hidden(data.size()[0])
		
		encoder_output1, encoder_hidden = E(torch.cat((decoder_padding,feat),2), encoder_hidden)
		decoder_result, decoder_output1, decoder_embed , decoder_hidden = D(torch.cat((decoder_padding,encoder_output1),2),decoder_hidden)
		
		attention = torch.transpose(decoder_output1,1,2)
		attention = E.attention(attention)
		attention = torch.sum(torch.transpose(attention,1,2),1,keepdim = True)
		#BOS token
		embed = Variable(torch.ones(data.size()[0],1).long().cuda())
		embed = D.embedding(embed)
		## stage 2 ##
		loss = 0
		encoder_padding = Variable(torch.zeros(data.size()[0],1,4096).cuda())

		for i in range(caption_index.size()[1]):
			encoder_output2, encoder_hidden = E(torch.cat((attention,encoder_padding),2),encoder_hidden)
			decoder_result2, decoder_output2, embed, decoder_hidden = D(torch.cat((embed,encoder_output2),2),decoder_hidden)
			if random.uniform(0,1) < SS_RATIO:
				embed = D.embedding(target[:,i]).view(-1,1,hidden_size)
			loss += criterion(decoder_result2,target[:,i])	

		total_loss += loss.data[0]
		loss.backward()
	
		torch.nn.utils.clip_grad_norm(E.parameters(), 0.25)
		torch.nn.utils.clip_grad_norm(D.parameters(), 0.25)
	
		e_optim.step()
		d_optim.step()
	print('Epoch {} Loss: {}'.format(epoch, total_loss/len(DS)))
	if epoch%SAVE == 0:
		torch.save({'encoder': E.state_dict(),'decoder': D.state_dict(), 'loss': total_loss/len(DS)}, os.path.join(directory, '{}.tar'.format(epoch)))
print("Finish training")
'''
### Testing Stage ###
TS = Testset(TEST_DIR,ID_PATH)
PS = Testset(PEER_DIR,PEER_ID_PATH)
CHECK_PATH = '121.tar'
checkpoint = torch.load(CHECK_PATH)
E = Encoder(4096+hidden_size,hidden_size)
E.load_state_dict(checkpoint['encoder'])
D = Decoder(2*hidden_size,hidden_size)
D.load_state_dict(checkpoint['decoder'])
E = E.cuda()
D = D.cuda()
E.eval()
D.eval()
MAX_LENGTH = 50

def output_sen(index_list,V):
	output = ""
	for i in index_list:
		if i == EOS_TOKEN:
			#output += " ."
			break
		elif i == BOS_TOKEN or i == PAD_TOKEN:
			continue
		else:
			output += V.index2word[i]+" "
	return output	
### Test ###
with open(sys.argv[2],'w') as f:
	for data in TS:
		caption_id = data[1]
		data = data[0]
		feat = Variable(torch.from_numpy(data).view(1,-1,4096).float()).cuda()
		encoder_hidden = E.init_hidden(1)
		decoder_hidden = D.init_hidden(1)
		decoder_padding = Variable(torch.zeros(1,feat.size()[1],hidden_size).cuda())
		encoder_output1, encoder_hidden = E(torch.cat((decoder_padding,feat),2),encoder_hidden)
		decoder_result1, decoder_output1,decoder_embed,  decoder_hidden = D(torch.cat((decoder_padding,encoder_output1),2),decoder_hidden)
			
		attention = torch.transpose(decoder_output1,1,2)
		attention = E.attention(attention)
		attention = torch.sum(torch.transpose(attention,1,2),1,keepdim = True)
		#BOS token
		embed = Variable(torch.ones(1,1).long().cuda())
		embed = D.embedding(embed)
			
		encoder_padding = Variable(torch.zeros(1,1,4096).cuda())
		result = []
		for i in range(MAX_LENGTH):
			encoder_output2, encoder_hidden = E(torch.cat((attention,encoder_padding),2), encoder_hidden)
			decoder_result2, decoder_output2, embed, decoder_hidden = D(torch.cat((embed,encoder_output2),2),decoder_hidden)
			top2 =  torch.topk(decoder_result2,k = 2, dim = 1)[1].data[0]
			if top2[0] == UNK_TOKEN:
				result.append(top2[1])
			else:
				result.append(top2[0])
	
		f.write('{},{}\n'.format(caption_id,output_sen(result,V)))
### Peer review ###
with open(sys.argv[3],'w') as f_p:
	for data in PS:
		caption_id = data[1]
		data = data[0]
		feat = Variable(torch.from_numpy(data).view(1,-1,4096).float()).cuda()
		encoder_hidden = E.init_hidden(1)
		decoder_hidden = D.init_hidden(1)
		decoder_padding = Variable(torch.zeros(1,feat.size()[1],hidden_size).cuda())
		encoder_output1, encoder_hidden = E(torch.cat((decoder_padding,feat),2),encoder_hidden)
		decoder_result1, decoder_output1,decoder_embed,  decoder_hidden = D(torch.cat((decoder_padding,encoder_output1),2),decoder_hidden)
			
		attention = torch.transpose(decoder_output1,1,2)
		attention = E.attention(attention)
		attention = torch.sum(torch.transpose(attention,1,2),1,keepdim = True)
		#BOS token
		embed = Variable(torch.ones(1,1).long().cuda())
		embed = D.embedding(embed)
			
		encoder_padding = Variable(torch.zeros(1,1,4096).cuda())
		result = []
		for i in range(MAX_LENGTH):
			encoder_output2, encoder_hidden = E(torch.cat((attention,encoder_padding),2), encoder_hidden)
			decoder_result2, decoder_output2, embed, decoder_hidden = D(torch.cat((embed,encoder_output2),2),decoder_hidden)
			top2 =  torch.topk(decoder_result2,k = 2, dim = 1)[1].data[0]
			if top2[0] == UNK_TOKEN:
				result.append(top2[1])
			else:
				result.append(top2[0])
		f_p.write('{},{}\n'.format(caption_id,output_sen(result,V)))

