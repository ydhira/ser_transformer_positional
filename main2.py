import numpy as np
import os, sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data.dataloader as dataloader
import torch.nn.utils.rnn as rnn
from tqdm import tqdm
import time
import string
# from utils import Dataset, multi_attention_model, cuda, self_attn_model, emotions_used
from new_utils import Dataset, cuda, self_attn_model, emotions_used, phoneme_class, Net
# from pytorch_pretrained_bert import *
from bert_embedding import BertEmbedding
from senet import se_resnet50 

# os.environ['DATA_PATH'] = '/home/hyd/multi-hop-attention/prepro_data'
os.environ['DATA_PATH'] = '/home/hyd/workhorse2/multi-hop-attention/prepro_data_8classes/' # accidently write 4 clases data into 8 classes
BATCH_SIZE = 16
n_epochs = 12
# MODEL_NAME = "64dim"
# MODEL_NAME = "8classes64dim"
model_type = "se_resnet50"
MODEL_NAME = "se_resnet50"#+phoneme_class
# emotions_used = ['ang', 'hap','sad', 'neu'] #previosuly used emotion classes 
# emotions_used = ['ang', 'hap', 'fru', 'sur', 'fea', 'exc', 'dis', 'sad', 'neu']


bert_embedding = BertEmbedding()
speakers={0:'Ses05_M',1:'Ses01_F',2:'Ses05_F',3:'Ses01_M',4:'Ses03_M',5:'Ses02_M',6:'Ses02_F',7:'Ses04_M',8:'Ses03_F',9:'Ses04_F'}

class IEMOCAP():

	def __init__(self, train_idx, test_idx, dev_idx):
		self.dev_set = None
		self.train_set = None
		self.test_set = None
		self.train_idx = train_idx
		self.test_idx = test_idx
		self.dev_idx = dev_idx

	def dev(self):
		if self.dev_set is None:
			self.dev_set = load_raw(os.environ['DATA_PATH'], self.dev_idx)
		return self.dev_set

	def train(self):
		if self.train_set is None:
			self.train_set = load_raw(os.environ['DATA_PATH'], self.train_idx)
		return self.train_set

	def test(self):
		if self.test_set is None:
			self.test_set = load_raw(os.environ['DATA_PATH'], self.test_idx)
		return self.test_set

def load_raw(path, idx):
	X, Y, Y_trans, Y_vad, Y_phone = [], [], [], [], [] 

	for i in idx:
		temp_set_X = np.load(os.path.join(path, "X_"+str(i)+".npy"), allow_pickle=True)
		temp_set_Y = np.load(os.path.join(path, "Y_emo"+str(i)+".npy"), allow_pickle=True)
		temp_set_Y_trans = np.load(os.path.join(path, "Y_trans_ctm"+str(i)+".npy"), allow_pickle=True) # using ctm trans 
		temp_set_Y_vad = np.load(os.path.join(path, "Y_vad"+str(i)+".npy"), allow_pickle=True)
		temp_set_Y_phone = np.load(os.path.join(path, "Y_phone"+str(i)+".npy"), allow_pickle=True)

		X.append(temp_set_X)
		Y.append(temp_set_Y)
		Y_trans.append(temp_set_Y_trans)
		Y_vad.append(temp_set_Y_vad)
		Y_phone.append(temp_set_Y_phone)

	X = np.concatenate(X)#[:num_ins]
	Y = np.concatenate(Y)#[:num_ins]
	Y_trans = np.concatenate(Y_trans)
	# print(Y_trans)
	# exit()
	Y_vad = np.concatenate(Y_vad)
	Y_phone = np.concatenate(Y_phone)
	return (X, Y, Y_trans, Y_vad, Y_phone)

def collate_lines(seq_list):
	seq_list = (j for j in seq_list if j is not None)
	inputs, targets, trans, vad, phone_target = zip(*seq_list)
	lens = [len(seq) for seq in inputs] #lens of each in input
	seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True) # Order of the sorting to sort others
	inputs = [inputs[i] for i in seq_order] #sort the inputs
	targets = [targets[i] for i in seq_order] #sort the targets
	trans = [trans[i] for i in seq_order] #sort the trans
	vad = [vad[i] for i in seq_order]
	embeds = bert_embedding(trans)
	embeds = list(map(lambda x: x[1], embeds))
	embeds = list(map(lambda x: torch.Tensor(x), embeds))
	trans = embeds
	return inputs, targets, trans, vad, None, phone_target

def val_model_trans(model, loader):
	model.eval()
	num_classes = len(emotions_used)
	cm = np.zeros(shape=(num_classes, num_classes))
	attention, phone_segmented, emo_target = [], [] , []
	with torch.no_grad():
		# model.to(device)
		total_loop = 0
		correct = 0

		for x1, target, x2, vad_target, _, phone_target in tqdm(loader):

			if len(x1) < 2:
				continue
			target = torch.LongTensor(target).cuda() if cuda else torch.LongTensor(target)
			vad_target = torch.FloatTensor(vad_target).cuda() if cuda else torch.FloatTensor(vad_target)
			if model_type == "attn":
				logits, logits_vad, _, attns= model(x1, x2)
			
				vad_target = torch.round(vad_target * 2) / 2
				loss2_V = F.mse_loss(logits_vad[:,0], vad_target[:,0])
				loss2_A = F.mse_loss(logits_vad[:,1], vad_target[:,1])
				loss2_D = F.mse_loss(logits_vad[:,2], vad_target[:,2])
				attention.append(attns)

			if model_type == "5cnn" or model_type == "se_resnet50":
				logits = model(x1)
				loss2_V, loss2_A, loss2_D, attention = None, None, None, None

			pred = torch.max(logits, dim=1)[1]
			correct += pred.eq(target).sum()

			total_loop += len(x1)
			
			phone_segmented.extend(phone_target)
			emo_target.extend(target)

			# print(logits_vad, vad_target)
			for i in range(len(target)):
				cm[target[i]][pred[i]] += 1

		if not isinstance(correct, int) :
			correct = correct.item()
		perc = correct / total_loop
		print("---- DEV/TEST ----")
		print("Correct: %.2f, Total: %.2f, Perc:  %.2f" %(correct, total_loop, perc ))
		if model_type == "attn":
			print("MSE V: %.3f, A: %.3f, D: %.3f" %(loss2_V, loss2_A, loss2_D))
		print(cm)
		return  loss2_V, loss2_A, loss2_D, perc, attention, phone_segmented, emo_target #torch.cat(attention, dim=0).cpu()


def train_epoch_trans(model, train_loader, optimizer):
	model.train()
	running_loss = 0.0
	start_time = time.time()
	c = 0
	correct, total_loop = 0, 0
	attention, phone_segmented, emo_target = [], [] , []
	# vad is a tuple of three = (V, A, D)
	for x1, target, x2, vad_target, _, phone_target in tqdm(train_loader): # (max_len x bs x dim) # (max_len x bs) #bs #bs
		# print(len(x1), len(x2))
		if len(x1) < 2:
			continue
		optimizer.zero_grad()
		# trans_target = torch.LongTensor(trans_target).cuda()
		target = torch.LongTensor(target).cuda() if cuda else torch.LongTensor(target)
		vad_target = torch.FloatTensor(vad_target).cuda() if cuda else torch.FloatTensor(vad_target)
		if model_type == "attn":
			logits, logits_vad, _, attns = model(x1, x2)

		if model_type == "5cnn" or model_type == "se_resnet50":
			logits = model(x1)

		pred = torch.max(logits, dim=1)[1]
		correct += pred.eq(target).sum()
		total_loop += len(x1)

		loss1 = F.nll_loss(logits, target, weight=class_balancing_weights)
		
		if model_type == "attn":
			loss2_V = F.mse_loss(logits_vad[:,0], vad_target[:,0])
			loss2_A = F.mse_loss(logits_vad[:,1], vad_target[:,1])
			loss2_D = F.mse_loss(logits_vad[:,2], vad_target[:,2])
			total_loss = loss1 + loss2_V + loss2_A + loss2_D#. Not using the VAD but only the cross entropy loss
			attention.append(attns)

		if model_type == "5cnn" or model_type == "se_resnet50":
			total_loss = loss1
			loss2_V, loss2_A, loss2_D, attention = None , None, None, None

		total_loss.backward()
		optimizer.step()
		running_loss += (total_loss.item())
		c+=1
		# print(type(attns[0][0]))
		phone_segmented.extend(phone_target) 
		emo_target.extend(target)
		# if (c>3):
			# break
	end_time = time.time()
	running_loss /= len(train_loader)
	print("---- TRAIN ----")
	print(correct)
	if not isinstance(correct, int) :
		correct = correct.item()
	running_acc = correct / total_loop
	print("class_loss {%.3f}, total_loss {%.3f}, Time: {%.3f} s" %(loss1, total_loss, end_time - start_time))
	if model_type == "attn":
		print("MSE V: %.3f, A: %.3f, D: %.3f" %(loss2_V, loss2_A, loss2_D))
	print('Training  class_acc: ', running_acc)

	return loss1, loss2_V, loss2_A, loss2_D, running_acc, attention, phone_segmented, emo_target


if __name__ == '__main__':
	idx = np.array([0,1,2,3,4,5,6,7,8,9])
	train_loss_l, train_acc_l = [], []
	dev_acc_l, test_acc_l = [], [] 
	dev_loss2_Vl, dev_loss2_Al, dev_loss2_Dl, = [], [], []
	test_loss2_Vl, test_loss2_Al, test_loss2_Dl, = [], [], []

	for j in range(len(idx)):
		
		print("J: ", j )
		test_idx = [idx[j-1]]
		dev_idx = [idx[j-1]]
		train_idx = np.setdiff1d(idx, test_idx + dev_idx)

		loader = IEMOCAP(train_idx, test_idx, dev_idx)
		trainX, trainY, trainTY, trainVAD, trainPhone = loader.train()
		devX, devY, devTY, devVAD, devPhone = loader.dev()
		testX, testY, testTY, testVAD, testPhone = loader.test()
		classn = len(set(trainY))
		print(trainY[0], trainVAD[0])
		unique, counts = np.unique(trainY, return_counts=True)
		print(unique, counts)
		print(counts/sum(counts))
		class_balancing_weights = counts/sum(counts)
		
		class_balancing_weights = (1 - (counts/sum(counts))) / (1 - (counts/sum(counts))).sum()
		class_balancing_weights_corr = []
		for idx_uni in unique:
			class_balancing_weights_corr.append(\
				class_balancing_weights[emotions_used.index(idx_uni)])
		class_balancing_weights = torch.tensor(class_balancing_weights_corr).cuda() if cuda \
								else torch.tensor(class_balancing_weights_corr)
		print(class_balancing_weights)
		print("No of classes %d " %(classn))

		if model_type == "attn":
			model = self_attn_model( 1, 4, 16, 64, 4, target_vocab)
		if model_type == "5cnn":
			model = Net(1, 4, 16, 64, 256, class_n=classn)
		if model_type == "se_resnet50":
			model = se_resnet50(num_classes=classn)

		device = torch.device("cuda:0" if cuda else "cpu")
		model = model.to(device).cuda() if cuda else model.to(device)

		devTY = list(map(lambda sentence: sentence.replace("\n", "").lower().translate(str.maketrans('','',string.punctuation)) , devTY))
		trainTY = list(map(lambda sentence: sentence.replace("\n", "").lower().translate(str.maketrans('','',string.punctuation)) , trainTY))
		testTY = list(map(lambda sentence: sentence.replace("\n", "").lower().translate(str.maketrans('','',string.punctuation)) , testTY))

		dev_vocab =  list(set(np.concatenate(list(map(lambda x: x.split(' '), devTY)))))
		train_vocab = list(set(np.concatenate(list(map(lambda x: x.split(' '), trainTY)))))
		target_vocab = list(set(np.concatenate((dev_vocab, train_vocab))))

		train = Dataset(trainX, trainY, trainTY, trainVAD, trainPhone)
		dev = Dataset(devX, devY, devTY, devVAD, devPhone)
		test = Dataset(testX, testY, testTY, testVAD, testPhone, test=True)

		print(trainX.shape, devX.shape, testX.shape)

		dataloader_args = dict(shuffle=True, batch_size=BATCH_SIZE, collate_fn = collate_lines, num_workers=0) 

		train_loader = dataloader.DataLoader(train, **dataloader_args)
		dev_loader = dataloader.DataLoader(dev, **dataloader_args)
		dataloader_args = dict(shuffle=False, batch_size=BATCH_SIZE, collate_fn = collate_lines, num_workers=0)
		test_loader = dataloader.DataLoader(test, **dataloader_args)

		optimizer = optim.Adam(model.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, threshold=1e-1, threshold_mode='abs')
		

		for i in range(n_epochs):
			print("Starting Training Epoch %d " % i)
			train_loss1, train_loss2_V, train_loss2_A, train_loss2_D, train_acc, train_attention, train_phone_segmented, train_emo_target \
								 =  train_epoch_trans(model, train_loader, optimizer)
			dev_loss2_V, dev_loss2_A, dev_loss2_D, dev_perc, dev_attention, dev_phone_segmented, dev_emo_target = val_model_trans(model, dev_loader)
			scheduler.step(dev_perc)

			if i == (n_epochs-1):
				train_loss_l.append(train_loss1)
				train_acc_l.append(train_acc)
				dev_acc_l.append(dev_perc)

				dev_loss2_Vl.append(dev_loss2_V)
				dev_loss2_Al.append(dev_loss2_A)
				dev_loss2_Dl.append(dev_loss2_D)

			fname = "models/"+str(j)+"_"+str(i)+MODEL_NAME+".model"
			torch.save({'epoch': i, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': dev_loss2_V, 'acc': dev_perc}, fname)


	print(train_loss_l)
	print(train_acc_l)
	print(dev_acc_l)
	# print(test_acc_l)
	print(dev_loss2_Vl)
	print(dev_loss2_Al)
	print(dev_loss2_Dl)