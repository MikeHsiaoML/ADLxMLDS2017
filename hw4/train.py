import os
import numpy as np
import scipy.misc
import pickle
import random
from config import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm

if torch.cuda.is_available():
	print("Using Cuda Device...")

def img_prepro(image_path):
	import skimage
	import skimage.io
	import skimage.transform
	img = skimage.io.imread(image_path)
	img_resized = skimage.transform.resize(img,(64,64),mode = 'constant')
	return img_resized

def image_augmentation(img, id_tag_list, img_path):
	inf = open('pretrained/all_tag_dict.pkl','rb')
	all_tag_dict = pickle.load(inf)

	split_ratio = 0.3
	for id_tag_pair in id_tag_list:
		id = id_tag_pair[0]
		tag = id_tag_pair[1][0]
		emb = id_tag_pair[1][1]
		tags = tag.strip().split()
		if len(tags) == 4:
			tag1 = ' '.join(tags[0:2])
			tag2 = ' '.join(tags[2:4])
			if random.random() < split_ratio:
				img.append(os.path.join(img_path, str(id)+'.jpg'))
				img.append(os.path.join(img_path, str(id)+'.jpg'))
				id_tag_list.append((id,(tag1,np.array(all_tag_dict[tag1],dtype = float))))
				id_tag_list.append((id,(tag2,np.array(all_tag_dict[tag2],dtype = float))))	
		#scipy.misc.imsave('{}.jpg'.format(tag),img_prepro(os.path.join(img_path,str(id)+'.jpg')))
	return img,id_tag_list

class text2img_Dataset(Dataset):
	def __init__(self, img_path,id_tag_dict):
		print("Loading descriptions about images...")
		read_file = open(id_tag_dict,'rb')
		self.id_tag_dict = pickle.load(read_file)
		self.id_tag_list = sorted(self.id_tag_dict.items(), key = lambda x: x[0])
		print("Loading images...")
		self.img = []
		for img in sorted([f for f in os.listdir(img_path)], key = lambda x: int(x.split('.')[0])):
			if int(img.split('.')[0]) in self.id_tag_dict.keys():
				self.img.append(os.path.join(img_path,img))
		print("Finish, there are {} images".format(len(self.img)))
		assert(len(self.img) == len(self.id_tag_list))
		print("Augmenting images...")
		
		self.img,self.id_tag_list = image_augmentation(self.img, self.id_tag_list, img_path)
		assert(len(self.img) == len(self.id_tag_list))
		print("There are {} images after augmentation".format(len(self.img)))
	
	def __len__(self):
		return len(self.img)

	def __getitem__(self,idx):
		check = False
		while not check:
			wrong = random.randrange(0,len(self.img))
			right_tag = self.id_tag_list[idx][1][0]
			wrong_tag = self.id_tag_list[wrong][1][0]
			if right_tag != wrong_tag:
				check = True
		#wrong = random.randrange(0,len(self.img))
		sample = {'right_img':img_prepro(self.img[idx]),
			'right_id_tag':self.id_tag_list[idx],
			'wrong_img':img_prepro(self.img[wrong]),
			'wrong_emb':self.id_tag_list[wrong][1][1]}
		return sample

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		# input is Z, going into a convolution
		self.deconv1 = nn.ConvTranspose2d(NOISE_DIM + FEAT_DIM, 64*8, 4, 1, 0, bias=False)
		self.bn1 = nn.BatchNorm2d(64*8)
		# state size. (ngf*8) x 4 x 4
		self.deconv2 = nn.ConvTranspose2d(64*8, 32*8, 4, 2, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(32*8)
		# state size. (ngf*4) x 8 x 8
		self.deconv3 = nn.ConvTranspose2d(32*8, 16*8, 4, 2, 1, bias=False)
		self.bn3 = nn.BatchNorm2d(16*8)
		# state size. (ngf*2) x 16 x 16
		self.deconv4 = nn.ConvTranspose2d(16*8, 8*8, 4, 2, 1, bias=False)
		self.bn4 = nn.BatchNorm2d(8*8)
		# state size. (ngf) x 32 x 32
		self.deconv5 = nn.ConvTranspose2d(8*8, 3, 4, 2, 1, bias=False)
		# state size. (nc) x 64 x 64
		self.emb2emb = nn.Linear(EMB_DIM,FEAT_DIM)

	def forward(self, noise, tags):
		tags = self.emb2emb(tags)
		input = torch.cat((noise,tags),1).unsqueeze(-1).unsqueeze(-1)
		out = F.relu(self.bn1(self.deconv1(input)),True)
		out = F.relu(self.bn2(self.deconv2(out)),True)
		out = F.relu(self.bn3(self.deconv3(out)),True)
		out = F.relu(self.bn4(self.deconv4(out)),True)
		out = F.tanh(self.deconv5(out))
		return out

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		# input is (nc) x 64 x 64
		self.conv1 = nn.Conv2d(3, 8*8, 4, 2, 1, bias=False)
		self.conv2 = nn.Conv2d(8*8, 16*8, 4, 2, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(16*8)
		# state size. (ndf*2) x 16 x 16
		self.conv3 = nn.Conv2d(16*8, 32*8, 4, 2, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(32*8)
		# state size. (ndf*4) x 8 x 8
		self.conv4 = nn.Conv2d(32*8, 64*8, 4, 2, 1, bias=False)
		self.bn3 = nn.BatchNorm2d(64*8)
		# state size. (ndf*8) x 4 x 4
		self.conv5 = nn.Conv2d(FEAT_DIM+64*8, 1, 4, 1, 0, bias=False)
		self.projection = nn.Sequential(
			nn.Linear(EMB_DIM, FEAT_DIM),
			nn.BatchNorm1d(FEAT_DIM),
			nn.LeakyReLU(0.2, inplace=True)
		)
	def forward(self, img, texts):
		out = F.leaky_relu(self.conv1(img),0.2,inplace = True)
		out = F.leaky_relu(self.bn1(self.conv2(out)),0.2, inplace = True)
		out = F.leaky_relu(self.bn2(self.conv3(out)),0.2, inplace = True)
		activation = F.leaky_relu(self.bn3(self.conv4(out)),0.2, inplace = True)
		texts = self.projection(texts).unsqueeze(-1).unsqueeze(-1)
		texts = texts.expand((texts.size(0),FEAT_DIM,4,4))
		out = torch.cat((activation,texts),1)
		out = F.sigmoid(self.conv5(out))
		return out.view(out.size(0)),activation

def train(dataset):
	l1_coef = 50
	l2_coef = 100
	print("Building model...")
	G = Generator().cuda()
	D = Discriminator().cuda()
	criterion = nn.BCELoss().cuda()
	l2loss = nn.MSELoss().cuda()
	l1loss = nn.L1Loss().cuda()
	print("Building Optimizer...")
	G_opt = optim.Adam(G.parameters(),lr = 0.0002, betas = (0.5, 0.999))
	D_opt = optim.Adam(D.parameters(),lr = 0.0002, betas = (0.5, 0.999))
	dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)
	for e in range(EPOCH):
		print("Epoch {}: ".format(e))
		D_print_loss = 0
		G_print_loss = 0
		for sample in tqdm(iter(dataloader)):
			batch_size = len(sample['right_img'])
			batch_img = sample['right_img']
			batch_list = sample['right_id_tag']
			batch_wrong_img = sample['wrong_img']
			batch_wrong_emb = sample['wrong_emb']
			for d_step in range(D_STEPS):
				D.zero_grad()
				## Real data for D
				#print("Real data")
				batch_img = Variable(batch_img.permute(0,3,1,2).float().cuda())
				D_real_decision,_ = D(batch_img,Variable(batch_list[1][1].float().cuda()))
				right_label = torch.FloatTensor([0.9 for i in range(batch_size)]).cuda()
				D_real_loss = criterion(D_real_decision, Variable(right_label))	
				## Real image, wrong text for D
				#print("Real image, wrong text")
				D_wrong_text_decision,_ = D(batch_img, Variable(batch_wrong_emb.float().cuda()))
				D_wrong_text_loss = criterion(D_wrong_text_decision, Variable(torch.zeros(batch_size).cuda()))
				## Fake image, right text for D
				#print("Fake image, right text")
				noise = Variable(torch.randn(batch_size, NOISE_DIM)).cuda()
				fake_img = G(noise, Variable(batch_list[1][1].float().cuda())).detach()
				D_fake_img_decision,_ = D(fake_img, Variable(batch_list[1][1].float().cuda()))
				D_fake_img_loss = criterion(D_fake_img_decision, Variable(torch.zeros(batch_size).cuda()))	

				## Wrong image, right text for D
				wrong_img = Variable(batch_wrong_img.permute(0,3,1,2).float().cuda())
				D_wrong_img_decision,_ = D(wrong_img,Variable(batch_list[1][1].float().cuda()))
				D_wrong_img_loss = criterion(D_wrong_img_decision, Variable(torch.zeros(batch_size).cuda()))	

				D_loss = D_real_loss+(D_wrong_text_loss+D_fake_img_loss+D_wrong_img_loss)/3
				D_loss.backward()

				D_print_loss += D_real_loss.data[0]+(D_wrong_text_loss.data[0]+\
						D_fake_img_loss.data[0]+D_wrong_img_loss.data[0])/3
				#print("Discirminator loss: {}".format((D_real_loss.data[0]+D_wrong_text_loss.data[0]+\
				#					D_fake_img_loss.data[0]+D_wrong_img_loss.data[0])/4))
				D_opt.step()
		
			for g_step in range(G_STEPS):
				G.zero_grad()
				## Train Generator
				#print("Generate image")
				noise = Variable(torch.randn(batch_size, NOISE_DIM)).cuda()
				fake_img = G(noise, Variable(batch_list[1][1].float().cuda()))
	
				D_G_img_decision,activation_fake = D(fake_img, Variable(batch_list[1][1].float().cuda()))
				_,activation_real = D(batch_img, Variable(batch_list[1][1].float().cuda()))

				activation_fake = torch.mean(activation_fake,0)
				activation_real = torch.mean(activation_real,0)
				G_loss = criterion(D_G_img_decision,Variable(torch.ones(batch_size).cuda()))\
					+l2_coef*l2loss(activation_fake, activation_real.detach())\
					+l1_coef*l1loss(fake_img, batch_img)

				G_loss.backward()
				G_print_loss += G_loss.data[0]
				#print("Generator loss: {}".format(G_loss.data[0]))
				G_opt.step()

		print("D loss: {}, G loss: {}".format(D_print_loss/len(dataloader), G_print_loss/G_STEPS/len(dataloader)))
		torch.save({'Generator': G.state_dict(), 'Discriminator': D.state_dict(), 'D_loss':D_print_loss/4/len(dataloader)\
				,'G_loss': G_print_loss/G_STEPS/len(dataloader), 'D_opt':D_opt.state_dict(), \
				'G_opt':G_opt.state_dict()},os.path.join(SAVE_DIR,"{}.tar".format(e)))				
def main():
	dataset = text2img_Dataset('./faces','./pretrained/id_tag_dict_aug.pkl')
	train(dataset)
if __name__ == "__main__":
	main()



