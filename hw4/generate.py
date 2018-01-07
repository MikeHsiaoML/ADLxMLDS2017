import os
import sys
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
from train import Generator, Discriminator,text2img_Dataset
torch.manual_seed(48)

l = 150
num_img = 5
checkpoint = torch.load("{}.tar".format(l))
G = Generator().cuda()
G.load_state_dict(checkpoint['Generator'])
G.eval()
inf = open('all_tag_dict.pkl','rb')
all_tag = pickle.load(inf)
inf.close()

with open(sys.argv[1]) as f:
	for l in f:
		index = l.strip().split(',')[0]
		tag = l.strip().split(',')[1]
		specified_tag = all_tag[tag]
		specified_tag = Variable(torch.from_numpy(specified_tag).repeat(num_img,1).cuda())
		z = Variable(torch.randn(num_img,NOISE_DIM).cuda())
		for i in range(num_img):
			img = G(z,specified_tag)[i].permute(1,2,0).data.cpu().numpy()
			scipy.misc.imsave('./samples/sample_{}_{}.jpg'.format(index,i+1),img)

