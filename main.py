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
from new_utils import Dataset, cuda, self_attn_model, emotions_used, phoneme_class
# from pytorch_pretrained_bert import *
from bert_embedding import BertEmbedding

# os.environ['DATA_PATH'] = '/home/hyd/multi-hop-attention/prepro_data'
os.environ['DATA_PATH'] = '/home/hyd/workhorse2/multi-hop-attention/prepro_data_8classes/' # accidently write 4 clases data into 8 classes
BATCH_SIZE = 16
n_epochs = 12
# MODEL_NAME = "64dim"
# MODEL_NAME = "8classes64dim"
MODEL_NAME = "self_attn_"+phoneme_class
# emotions_used = ['ang', 'hap','sad', 'neu'] #previosuly used emotion classes 
# emotions_used = ['ang', 'hap', 'fru', 'sur', 'fea', 'exc', 'dis', 'sad', 'neu']

## bert model ##
# print(BertConfig)
# print(dir(BertConfig))
# config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True,)
# config = BertConfig(output_hidden_states=True)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertModel.from_pretrained('bert-base-uncased')
# bert_model = bert_model.cuda()
# bert_model.config.output_hidden_states = True
# print(bert_model.config.output_hidden_states)
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
		temp_set_X = np.load(os.path.join(path, "X_"+str(i)+".npy"))
		temp_set_Y = np.load(os.path.join(path, "Y_emo"+str(i)+".npy"))
		temp_set_Y_trans = np.load(os.path.join(path, "Y_trans_ctm"+str(i)+".npy")) # using ctm trans 
		temp_set_Y_vad = np.load(os.path.join(path, "Y_vad"+str(i)+".npy"))
		temp_set_Y_phone = np.load(os.path.join(path, "Y_phone"+str(i)+".npy"))

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
	inputs, targets, trans, vad, indices = zip(*seq_list)
	lens = [len(seq) for seq in inputs] #lens of each in input
	seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True) # Order of the sorting to sort others
	inputs = [inputs[i] for i in seq_order] #sort the inputs
	targets = [targets[i] for i in seq_order] #sort the targets
	trans = [trans[i] for i in seq_order] #sort the trans
	vad = [vad[i] for i in seq_order]
	# print(trans[0], type(trans[0]))
	# print("**************")
	# print(trans)
	embeds = bert_embedding(trans)
	# print(embeds)
	# print(len(embeds))
	# print("*****************")
	# print(len(embeds[0]), len(embeds[1]))
	# exit()
	# print(embeds[0][0])
	# print(len(embeds[0][1]))
	# print(len(embeds[0][1][0]))
	# print(embeds[0][1][0].shape)
	embeds = list(map(lambda x: x[1], embeds))
	# print(len(embeds[0]))
	embeds = list(map(lambda x: torch.Tensor(x), embeds))
	# print(len(embeds[0]))
	# print(embeds[0].shape)
	trans = embeds
	# for t in trans: print(len(t))
	################# this is for pytorch-transformers #########
	# trans = [tokenizer.tokenize('[CLS] ' + sent + ' [SEP]') for sent in trans]
	# trans_tokens = [tokenizer.convert_tokens_to_ids(sent) for sent in trans]
	# results = torch.zeros((len(trans_tokens), bert_model.config.hidden_size))#.long()
	# results2 = []
	# output_hidden_states = None
	# batch_size_now = 1

	# with torch.no_grad():
	# 	for stidx in range(0, len(trans_tokens), batch_size_now):
	# 	    X = trans_tokens[stidx:stidx + batch_size_now]
	# 	    X = torch.LongTensor(X).cuda()
	# 	    sequence_output, pooled_output = bert_model(X)
	# 	    # print(pooled_output.shape) # (bs, 768)
	# 	    # print(sequence_output[0].shape, sequence_output[1].shape, sequence_output[2].shape) # (1, seq_len, 768)
	# 	    results[stidx:stidx + batch_size_now,:] = pooled_output.cpu()
	# 	    results2.append(torch.cat(sequence_output, dim=0).transpose(0, 1))
	# trans = (results, results2)
    ######################

	# trans_lens = [len(t) for t in trans]
	# trans_seq_order = sorted(range(len(trans_lens)), key=trans_lens.__getitem__, reverse=True) # Order of the sorting to sort others
	# trans_seq_order_new = []
	# for i in trans_seq_order:
	# 	# if trans_lens[i] > 0:
	# 	trans_seq_order_new.append(i)

	# trans_seq_order = trans_seq_order_new
	# trans_target =  [targets[i] for i in trans_seq_order]# sorting labels for transcripts 
	# trans = [trans[i] for i in trans_seq_order] #sort the trans 
	# print(trans)
	# print(np.array(trans_lens)[trans_seq_order])
	
	# print(len(inputs), len(targets), trans)
	return inputs, targets, trans, vad, None, indices

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
			# print(len(x1), len(x2))

			if len(x1) < 2:
				continue
			target = torch.LongTensor(target).cuda() if cuda else torch.LongTensor(target)
			vad_target = torch.FloatTensor(vad_target).cuda() if cuda else torch.FloatTensor(vad_target)
			logits, logits_vad, _, attns= model(x1, x2)
			pred = torch.max(logits, dim=1)[1]
			correct += pred.eq(target).sum()
			vad_target = torch.round(vad_target * 2) / 2
			loss2_V = F.mse_loss(logits_vad[:,0], vad_target[:,0])
			loss2_A = F.mse_loss(logits_vad[:,1], vad_target[:,1])
			loss2_D = F.mse_loss(logits_vad[:,2], vad_target[:,2])
			
			total_loop += len(x1)
			attention.append(attns)
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
		logits, logits_vad, _, attns = model(x1, x2)
		pred = torch.max(logits, dim=1)[1]
		# print(pred, trans_target)
		correct += pred.eq(target).sum()
		total_loop += len(x1)
		loss1 = F.nll_loss(logits, target, weight=class_balancing_weights)
		loss2_V = F.mse_loss(logits_vad[:,0], vad_target[:,0])
		loss2_A = F.mse_loss(logits_vad[:,1], vad_target[:,1])
		loss2_D = F.mse_loss(logits_vad[:,2], vad_target[:,2])
		# print(vad_target)
		# print(logits_vad)
		# print(loss2)
		# exit()
		total_loss = loss1 #+ loss2_V + loss2_A + loss2_D. Not using the VAD but only the cross entropy loss 
		total_loss.backward()
		optimizer.step()
		running_loss += (total_loss.item())
		c+=1
		print(type(attns[0][0]))
		attention.append(attns)
		phone_segmented.extend(phone_target) 
		emo_target.extend(target)
		if (c>3):
			break
	end_time = time.time()
	running_loss /= len(train_loader)
	print("---- TRAIN ----")
	print(correct)
	if not isinstance(correct, int) :
		correct = correct.item()
	running_acc = correct / total_loop
	print("class_loss {%.3f}, total_loss {%.3f}, Time: {%.3f} s" %(loss1, total_loss, end_time - start_time))
	print("MSE V: %.3f, A: %.3f, D: %.3f" %(loss2_V, loss2_A, loss2_D))
	# print('Training Loss: ', running_loss, 'Time: ', end_time - start_time, 's')
	print('Training  class_acc: ', running_acc)

	return loss1, loss2_V, loss2_A, loss2_D, running_acc, attention, phone_segmented, emo_target


if __name__ == '__main__':
	idx = np.array([0,1,2,3,4,5,6,7,8,9])
	train_loss_l, train_acc_l = [], []
	dev_acc_l, test_acc_l = [], [] 
	dev_loss2_Vl, dev_loss2_Al, dev_loss2_Dl, = [], [], []
	test_loss2_Vl, test_loss2_Al, test_loss2_Dl, = [], [], []

	for j in range(len(idx)):
		# j = 1
		print("J: ", j )
		test_idx = [idx[j-1]]
		# dev_idx = [idx[j]]
		dev_idx = [idx[j-1]]
		train_idx = np.setdiff1d(idx, test_idx + dev_idx)
		# train_idx = [0]#,1,2,3,4,5,6,7]
		# dev_idx = [8]
		# test_idx = [9]
		

		loader = IEMOCAP(train_idx, test_idx, dev_idx)
		trainX, trainY, trainTY, trainVAD, trainPhone = loader.train()
		devX, devY, devTY, devVAD, devPhone = loader.dev()
		testX, testY, testTY, testVAD, testPhone = loader.test()
		classn = len(set(trainY)) - 1 # removing the neu class

		print(trainY[0], trainVAD[0])

		unique, counts = np.unique(trainY, return_counts=True)
		print("Total number of unique classes {%d}, Total count {%d}" %(unique, counts))
		print(counts/sum(counts))
		class_balancing_weights = counts/sum(counts)
		
		class_balancing_weights = (1 - (counts/sum(counts))) / (1 - (counts/sum(counts))).sum()
		class_balancing_weights_corr = []
		for idx_uni in unique:
			class_balancing_weights_corr.append(\
				class_balancing_weights[emotions_used.index(idx_uni)])
		class_balancing_weights = torch.tensor(class_balancing_weights_corr).cuda() if cuda \
								else torch.tensor(class_balancing_weights_corr)
		# class_balancing_weights = np.zeros(4)
		# class_balancing_weights.fill(0.5)
		# class_balancing_weights[-1] =  0

		print(class_balancing_weights)
		# exit()
		print("No of classes %d " %(classn))
		# print(testTY[0], len(testTY))
		# vocab = trainTY.extend(devTY)

		devTY = list(map(lambda sentence: sentence.replace("\n", "").lower().translate(str.maketrans('','',string.punctuation)) , devTY))
		trainTY = list(map(lambda sentence: sentence.replace("\n", "").lower().translate(str.maketrans('','',string.punctuation)) , trainTY))
		testTY = list(map(lambda sentence: sentence.replace("\n", "").lower().translate(str.maketrans('','',string.punctuation)) , testTY))

		dev_vocab =  list(set(np.concatenate(list(map(lambda x: x.split(' '), devTY)))))
		# dev_vocab = list(map(lambda x: x.replace(" ", "").lower().translate(str.maketrans('','',string.punctuation)), dev_vocab))
		train_vocab = list(set(np.concatenate(list(map(lambda x: x.split(' '), trainTY)))))
		# train_vocab = list(map(lambda x: x.replace(" ", "").lower().translate(str.maketrans('','',string.punctuation)), train_vocab))
		target_vocab = list(set(np.concatenate((dev_vocab, train_vocab))))

		train = Dataset(trainX, trainY, trainTY, trainVAD, trainPhone)
		dev = Dataset(devX, devY, devTY, devVAD, devPhone)
		test = Dataset(testX, testY, testTY, testVAD, testPhone, test=True)

		print(trainX.shape, devX.shape, testX.shape)
		# model = CatEmotion(classn,128,3) # categorical emotion 
		
		# device = torch.device("cuda:0" if cuda else "cpu")
		# model = model.to(device).cuda()

		# model = TransEmotion(len(target_vocab), 128,3,classn+1) # transcription emotion
										# hidden_size, nlayers, class_n, vocab_size
		# print(len(target_vocab))
		# with open('target_vocab.txt', 'w') as fopen:
		# 	for word in target_vocab:
		# 		fopen.write(word+"\n")

		# exit()
		model = self_attn_model( 1, 4, 16, 64, 4, target_vocab)
		# model = multi_attention_model(768, 1, classn+1, target_vocab) # adding back the neutral class 
		device = torch.device("cuda:0" if cuda else "cpu")
		model = model.to(device).cuda() if cuda else model.to(device)

		dataloader_args = dict(shuffle=True, batch_size=BATCH_SIZE, collate_fn = collate_lines, num_workers=0) 
		# if cuda else dict(shuffle=True, batch_size=BATCH_SIZE) #131072
		train_loader = dataloader.DataLoader(train, **dataloader_args)
		dev_loader = dataloader.DataLoader(dev, **dataloader_args)
		dataloader_args = dict(shuffle=False, batch_size=BATCH_SIZE, collate_fn = collate_lines, num_workers=0)
		# if cuda else dict(shuffle=False, batch_size=BATCH_SIZE) #262144
		test_loader = dataloader.DataLoader(test, **dataloader_args)

		optimizer = optim.Adam(model.parameters(), lr=0.001)
		# optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1, momentum=0.9, weight_decay=0.0001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, threshold=1e-1, threshold_mode='abs')
		

		for i in range(n_epochs):
			print("Starting Training Epoch %d " % i)
			# train_loss, train_acc =  train_epoch(model, train_loader, optimizer)
			train_loss1, train_loss2_V, train_loss2_A, train_loss2_D, train_acc, train_attention, train_phone_segmented, train_emo_target \
								 =  train_epoch_trans(model, train_loader, optimizer)
			dev_loss2_V, dev_loss2_A, dev_loss2_D, dev_perc, dev_attention, dev_phone_segmented, dev_emo_target = val_model_trans(model, dev_loader)
			# test_loss2_V, test_loss2_A, test_loss2_D, test_perc = val_model_trans(model, test_loader)
			scheduler.step(dev_perc)
			# exit()
			if i == (n_epochs-1):
				train_loss_l.append(train_loss1)
				train_acc_l.append(train_acc)
				dev_acc_l.append(dev_perc)
				# test_acc_l.append(test_perc)

				dev_loss2_Vl.append(dev_loss2_V)
				dev_loss2_Al.append(dev_loss2_A)
				dev_loss2_Dl.append(dev_loss2_D)

				# test_loss2_Vl.append(test_loss2_V)
				# test_loss2_Al.append(test_loss2_A)
				# test_loss2_Dl.append(test_loss2_D)

			fname = "models/"+str(j)+"_"+str(i)+MODEL_NAME+".model"
			torch.save({'epoch': i, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': dev_loss2_V, 'acc': dev_perc}, fname)

		# np.save(phoneme_class+"/"+str(j)+"_train_attention" ,train_attention)
		# np.save(phoneme_class+"/"+str(j)+"_train_phone_segmented" ,train_phone_segmented)
		# np.save(phoneme_class+"/"+str(j)+"_train_emo_target", train_emo_target)

		# np.save(phoneme_class+"/"+str(j)+"_dev_attention" ,dev_attention)
		# np.save(phoneme_class+"/"+str(j)+"_dev_phone_segmented" ,dev_phone_segmented)
		# np.save(phoneme_class+"/"+str(j)+"_dev_emo_target", dev_emo_target)

		print(train_loss_l)
		print(train_acc_l)
		print(dev_acc_l)
		print(test_acc_l)
		print(dev_loss2_Vl)
		print(dev_loss2_Al)
		print(dev_loss2_Dl)

		# print(test_loss2_Vl)
		# print(test_loss2_Al)
		# print(test_loss2_Dl)
