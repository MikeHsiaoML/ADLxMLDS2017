from utils import *
import editdistance
from config import SAVE_DIR,USE_CUDA,PHONE_NUM 
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
import os
import numpy as np
import sys

class Feature_Dataset(Dataset):
	def __init__(self,data_dir, dataset,mode):
		DATA_DIR = data_dir
		print("Preparing dataset")
		self.dataset = dataset
		self.mode = mode
		if dataset != 'all':
			filename = os.path.join(DATA_DIR,dataset,mode+'.ark')
		else:
			filename = os.path.join(DATA_DIR,'mfcc',mode+'.ark')
			fbank_file = os.path.join(DATA_DIR,'fbank',mode+'.ark')

		self.feature = defaultdict(list)
		with open(filename,'r') as f:
			for line in f.readlines():
				info = line.strip().split()
				speaker = '_'.join(info[0].split('_')[:-1])
				self.feature[speaker].append(info[1:])	
		if dataset == 'all':
			with open(fbank_file,'r') as f:
				for line in f.readlines():
					info = line.strip().split()
					speaker = '_'.join(info[0].split('_')[:-1])
					self.feature[speaker].append(info[1:])
			for speaker,feature in self.feature.items():
				seq_len = int(len(self.feature[speaker])/2)
				for i in range(seq_len):
					self.feature[speaker][i] += self.feature[speaker][i+seq_len]
				del self.feature[speaker][seq_len:]
	def __len__(self):
		return len(self.feature)
	
	def __getitem__(self,index):
		chosen_speaker = list(self.feature.keys())[index]
		feature = torch.from_numpy(np.array(self.feature[chosen_speaker]).astype('float'))
		feature = feature.cuda() if USE_CUDA else feature
		return [chosen_speaker,feature]
    
class LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, layers=1,bi=False):
		super(LSTM, self).__init__()
		self.layers = layers	
		self.hidden_size = hidden_size
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, bidirectional=bi, dropout = 0.3)
		self.hidden2out = nn.Linear(hidden_size,PHONE_NUM)
		self.softmax = nn.LogSoftmax()
		self.bi = bi	
	def init_hidden(self):
		direction = 2 if self.bi else 1
		if USE_CUDA:
			return (Variable(torch.zeros(self.layers*direction, 1, self.hidden_size)).cuda(),\
				Variable(torch.zeros(self.layers*direction, 1, self.hidden_size)).cuda())
		else:
			return (Variable(torch.zeros(self.layers*direction, 1, self.hidden_size)),\
				Variable(torch.zeros(self.layers*direction, 1, self.hidden_size))) 
 
	def forward(self, input_seq, hidden):
		input_seq = input_seq.view(len(input_seq),1,-1)
		output, hidden = self.lstm(input_seq,hidden) 
		if self.bi:
			output = (output[:,:,:self.hidden_size]+output[:,:,self.hidden_size:])/2
		output = self.hidden2out(output.view(output.size()[0],-1))
		output = self.softmax(output)
		return output
    
class C_RNN(nn.Module):
	def __init__(self, group_size, input_size, hidden_size, layers=1,bi=False):
		super(C_RNN, self).__init__()
		self.group_size = group_size
		self.feature_len = 5 
		self.filter = 10 
		self.layers = layers	
		self.hidden_size = hidden_size
		self.cnn = nn.Conv2d(1,self.filter,kernel_size = (self.group_size,self.feature_len))
		self.pooling = nn.MaxPool2d((1,3))
		self.lstm = nn.LSTM(249, hidden_size, num_layers=layers, bidirectional=bi, dropout = 0.5)
		self.hidden2out = nn.Linear(hidden_size,PHONE_NUM)
		self.softmax = nn.LogSoftmax()
	
	def init_hidden(self):
		if USE_CUDA:
			return (Variable(torch.rand(self.layers, 1, self.hidden_size)).cuda(),\
				Variable(torch.rand(self.layers, 1, self.hidden_size)).cuda())
		else:
			return (Variable(torch.rand(self.layers, 1, self.hidden_size)),\
				Variable(torch.rand(self.layers, 1, self.hidden_size))) 
 
	def forward(self, input_seq, hidden):
		padding_size = int(self.group_size/2)
		input_seq = torch.cat((input_seq[0].repeat(padding_size,1),input_seq),0)
		input_seq = torch.cat((input_seq,input_seq[-1].repeat(padding_size,1)),0)
		for i in range(len(input_seq)-self.group_size+1):
			feature = input_seq[i:i+self.group_size,39:]
			feature = feature.contiguous().view(1,1,feature.size()[0],feature.size()[1])
			if i == 0:
				input_feature = self.cnn(feature)
				input_feature = self.pooling(input_feature)
				input_feature = input_feature.view(1,self.filter*input_feature.size()[-1])
			else:
				new_feature = self.cnn(feature)
				new_feature = self.pooling(new_feature)
				new_feature = new_feature.view(1,self.filter*new_feature.size()[-1])
				input_feature = torch.cat((input_feature,new_feature),0)
		
		input_feature = input_feature.view(input_feature.size()[0],-1)
		input_feature = torch.cat((input_feature,input_seq[padding_size:len(input_seq)-padding_size,:39]),1)
		input_feature = input_feature.view(input_feature.size()[0],1,-1)
		output, hidden = self.lstm(input_feature,hidden) 
		output = self.hidden2out(output.view(output.size()[0],-1))
		output = self.softmax(output)
		return output

def train(data_dir, feature,label, epochs, model, layer, hidden, save,postfix, index2char, index2phone, phone_map, phone2index):
	dataset = Feature_Dataset(data_dir, feature,'train')
	train_size = int(0.9*len(dataset))
	if feature == 'mfcc':
		feature_dim = 39
	elif feature == 'fbank':
		feature_dim = 69
	elif feature == 'all':
		feature_dim = 108

	print("Building model and optimizer...")
	if model == 'LSTM':
		train_model = LSTM(feature_dim,hidden,layer)
	elif model == 'C_RNN':
		group_size = 5 
		train_model = C_RNN(group_size,feature_dim,hidden,layer)
	elif model == 'BiLSTM':
		train_model = LSTM(feature_dim, hidden, layer, bi = True)
	
	if USE_CUDA:
		train_model = train_model.cuda()
	optimizer = optim.Adam(train_model.parameters(), lr = 0.0001)
	#optimizer = optim.SGD(train_model.parameters(),lr = 0.1)
	criterion = nn.NLLLoss()
	if USE_CUDA:
		criterion = criterion.cuda() 

	for epoch in range(1,epochs+1):
		print("Epoch {}".format(epoch))
		epoch_loss = 0
		epoch_edit = 0
		for i in tqdm(range(1,train_size+1)):
			data = dataset[i-1]
			speaker = data[0]
		
			train_model.zero_grad()
			input_hidden = train_model.init_hidden()
			
			train_feature = Variable(data[1].float())
			output =  train_model(train_feature,input_hidden)
			
			output_seq = test_trim(index2char, index2phone, phone_map, phone2index, torch.max(output,1)[1].data.cpu().numpy())
			target_seq = trim_and_map(index2char,index2phone, phone_map, phone2index, [[int(l)] for l in label[speaker]])
			
			target = Variable(torch.from_numpy(np.array(label[speaker]).astype('int')))
			target = target.cuda() if USE_CUDA else target
			
			loss = criterion(output,target)
			edit = editdistance.eval(output_seq,target_seq)

			epoch_loss += loss.data[0]/train_size
			epoch_edit += edit/train_size
		
			loss.backward()
			optimizer.step()

		print("Negative log-likelihood: {}".format(epoch_loss))
		print("Edit distance: {} ".format(epoch_edit))
		val_loss = 0
		val_edit = 0
		for i in tqdm(range(train_size+1,len(dataset)+1)):
			data = dataset[i-1]
			speaker = data[0]
			val_feature = Variable(data[1].float())
			
			output = train_model(val_feature,train_model.init_hidden())
			target = Variable(torch.from_numpy(np.array(label[speaker]).astype('int')))
			target = target.cuda() if USE_CUDA else target
			
			val_loss += criterion(output,target).data[0]		
			output_seq = test_trim(index2char,index2phone, phone_map, phone2index,torch.max(output,1)[1].data.cpu().numpy())
			target_seq = trim_and_map(index2char,index2phone, phone_map, phone2index,[[int(l)] for l in label[speaker]])
				
			val_edit += editdistance.eval(output_seq,target_seq)
		print("Validation loss: {}".format(val_loss/(len(dataset)-train_size)))
		print("Validation edit distance: {}".format(val_edit/(len(dataset)-train_size)))

		if epoch%save == 0:
			directory = os.path.join(SAVE_DIR, feature, model, '{}-{}{}'.format(layer,hidden,postfix))
			if not os.path.exists(directory):
				os.makedirs(directory)
			torch.save({
				'model': train_model.state_dict(),
                		'opt': optimizer.state_dict(),
                		'val_loss': val_loss/(len(dataset)-train_size),
				'val_edit': val_edit/(len(dataset)-train_size),
				}, os.path.join(directory, '{}.tar'.format(epoch)))
	print("Finish training")

def test(data_dir, test, feature, model, hidden, layer,  output, index2char, index2phone, phone_map, phone2index):
	print(test)
	ans = open(output,'w')
	ans.write('id,phone_sequence\n')
	test_set = Feature_Dataset(data_dir, feature,'test')
	if feature == 'mfcc':
		feature_dim = 39
	elif feature == 'fbank':
		feature_dim = 69
	elif feature == 'all':
		feature_dim = 108
	
	if model == 'LSTM':
		test_model = LSTM(feature_dim, hidden, layer)
	elif model == 'BiLSTM':
		test_model = LSTM(feature_dim,hidden,layer,bi = True)
	elif model == 'C_RNN':
		group_size = 5
		test_model = C_RNN(group_size, feature_dim, hidden, layer)    
	
	checkpoint = torch.load(test)
	test_model.load_state_dict(checkpoint['model'])
	test_model.eval()
	if USE_CUDA:
		test_model = test_model.cuda()		
	for i in tqdm(range(1,len(test_set)+1)):
		data = test_set[i-1]
		speaker = data[0]
		test_feature = Variable(data[1].float())
		test_hidden = test_model.init_hidden()
		output = torch.max(test_model(test_feature,test_hidden),1)[1]
		result = test_trim(index2char,index2phone, phone_map, phone2index, output.data.cpu().numpy())
		ans.write('{},{}\n'.format(speaker,result))
	ans.close()

def train_loss(train_dir):
    loss_dict = {}
    edit_dict = {}
    for f in os.listdir(train_dir):
        checkpoint = torch.load(os.path.join(train_dir,f))
        loss_dict[f] = checkpoint['val_loss']
        edit_dict[f] = checkpoint['val_edit']
    print(loss_dict)
    print(edit_dict)
	

def main(data_dir, output, mode):
	if USE_CUDA:
		print("Using cuda...")
	else:
		print("Suggest using cuda")
	
	test_model = 'cnn.tar'
	feature = 'all'
	model = 'C_RNN'
	hidden = 256
	layer = 2
	epochs = 30
	save = 1
	postfix = ''

	phone_map = make_phone_map(data_dir)
	phone2index,index2phone, index2char = make_phone_char(data_dir)
	label = make_label(phone2index,data_dir)
	
	if mode == 'test':
		test(data_dir, test_model, feature, model, hidden, layer, output, index2char, index2phone, phone_map, phone2index)
	elif mode == 'train':
		train(data_dir, feature, label,  epochs, model, layer, hidden, save, postfix, index2char, index2phone, phone_map, phone2index)


if __name__ == '__main__':
	data_dir = sys.argv[1]
	output = sys.argv[2]
	mode = sys.argv[3]
	main(data_dir, output, mode)    

