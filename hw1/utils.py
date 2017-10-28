from config import DATA_DIR
from collections import defaultdict
import os
import torch

def make_phone_map():
	phone_map = {}
	with open(os.path.join(DATA_DIR,'phones/48_39.map'),'r') as f:
		for line in f.readlines():
			m1 = line.strip().split('\t')[0]
			m2 = line.strip().split('\t')[1]
			phone_map[m1] = m2
	return phone_map

def make_phone_char():
	phone2index = {}
	index2phone = {}
	index2char = {}
	with open(os.path.join(DATA_DIR,'48phone_char.map'),'r') as f:
		for line in f.readlines():
			info = line.strip().split('\t')
			phone2index[info[0]] = info[1]
			index2phone[int(info[1])] = info[0]
			index2char[int(info[1])] = info[2]
	return phone2index,index2phone,index2char

def make_label(phone2index):
	label = defaultdict(list)
	with open(os.path.join(DATA_DIR,'label/train.lab'),'r') as f:
		for line in f.readlines():
			info = line.strip().split(',')
			speaker = '_'.join(info[0].split('_')[:-1])
			label[speaker].append(phone2index[info[1]])
	return label

def trim_and_map(index2char,index2phone, phone_map, phone2index, output):
    result = ''
    tmp = ''
    current = output[0][0]
    for i in range(len(output)):
        index = output[i][0]
        if phone_map[index2phone[index]] == 'sil':
            continue
        else:
            tmp += index2char[int(phone2index[phone_map[index2phone[index]]])]
    for i in range(len(tmp)):
        if len(result) > 0 and tmp[i] == result[-1]:
            continue
        else:
            result += tmp[i]
    return result

def test_trim(index2char,index2phone,phone_map, phone2index,output):
    result = ''
    tmp = ''
    current = output[0][0]
    for i in range(len(output)):
        index = output[i][0]
        if phone_map[index2phone[index]] == 'sil':
            continue
        else:
            tmp += index2char[int(phone2index[phone_map[index2phone[index]]])]
    count = 0
    for i in range(len(tmp)):
        if i > 0 and tmp[i] != tmp[i-1]:
            count = 1
            continue
        elif i > 0 and count >= 2:
            if len(result) > 0 and tmp[i] == result[-1]:
                continue
            else:
                result += tmp[i]
                count = 0 
        else:
            count += 1
    return result



def normalize(data):
    mean = torch.mean(data,0)
    mean = mean.repeat(data.size()[0],1)
    std = torch.std(data,0)+1e-8
    std = std.repeat(data.size()[0],1)
    return (data-mean)/std


