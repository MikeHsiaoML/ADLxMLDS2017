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
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, bidirectional=bi, dropout = 0.4)
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
	elif model == 'BiLSTM':
		train_model = LSTM(feature_dim, hidden, layer, bi = True)
	
	if USE_CUDA:
		train_model = train_model.cuda()
	optimizer = optim.Adam(train_model.parameters(), lr = 0.001)
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
			
			train_feature = Variable(normalize(data[1]).float())
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
			val_feature = Variable(normalize(data[1]).float())
			
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
	
	checkpoint = torch.load(test)
	test_model.load_state_dict(checkpoint['model'])
	test_model.eval()
	if USE_CUDA:
		test_model = test_model.cuda()		
	for i in tqdm(range(1,len(test_set)+1)):
		data = test_set[i-1]
		speaker = data[0]
		test_feature = Variable(normalize(data[1]).float())
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
	
	test_model = 'best.tar'
	feature = 'mfcc'
	model = 'BiLSTM'
	hidden = 512
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

