import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn
import pickle
import numpy as np 
# from bert_embedding import BertEmbedding
import sys, os, string
import re
from more_itertools import unique_everseen
# import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, classification_report
emotions_used = ['ang', 'hap','sad', 'neu'] # previosuly used emotion classes 
cuda = True 
os.environ['DATA_PATH'] = '/home/hyd/workhorse2/multi-hop-attention/prepro_data_4classes-mellog40/' 
NUM_OF_WORDS = 6

def get_metrics(pred, y_test, mode='train-'):
	# pred=pred.detach().numpy()
	# pred=np.argmax(pred,1)
	precision_metric = precision_score(y_test, pred, average = "macro")
	recall_metric = recall_score(y_test, pred, average = "macro")
	accuracy_metric = accuracy_score(y_test, pred)
	f1_metric = f1_score(y_test, pred, average = "macro")
	# roc_metric=roc_auc_score(y_test, pred, average = "macro")
	return {mode+"precision":precision_metric, mode+"recall": recall_metric, mode+"accuracy": accuracy_metric, mode+"f1": f1_metric}



def read_phonemes(filename="/home/hyd/workhorse2/multi-hop-attention/code/cmu_phones.txt"):
	d = {}
	with open(filename, "r") as f:
		lines=f.readlines()
		for line in lines:
			line_s=line.rsplit()
			phoneme, classn = line_s[0], line_s[1]
			val = d.get(classn, []) 
			val.append(phoneme)
			d[classn] = val
	return d

def  clean_mfc(mfc, phone_seg, phoneme_mapping ) :
	if phone_seg is None:
		return mfc 
	phone_seg.reverse()
	new_phone_seg = []
	# print(phoneme_mapping[phoneme_class])
	for ps in phone_seg:
		# print(ps[0])
		if ps[0] == "'SIL'": # for removing SIL frames
		# if ps[0].replace("'", "") not in phoneme_mapping[phoneme_class]:
			mfc = np.delete(mfc, np.s_[int(ps[1]):int(ps[2])+1], 0)
		else:
			new_phone_seg.append(ps)
	new_phone_seg.reverse()
	return mfc , new_phone_seg

def get_phoneme_word_dict():
	lines = open('/home/hyd/workhorse2/multi-hop-attention/data/IEMOCAP_full_release/Documentation/phonemes.txt', 'r').readlines() 
	phoneme_dict={}
	for i, l in enumerate(lines):
		ph = l.split(' ')[0].replace('\n', '')
		phoneme_dict[ph]=i

	lines = open('/home/hyd/workhorse2/multi-hop-attention/data/IEMOCAP_full_release/Documentation/corpus.dic', 'r').readlines() 
	word_dict={}
	for i, l in enumerate(lines):
		wd = l.split(' ')[0].lower().replace('\n', '')
		word_dict[wd]=i
	return phoneme_dict, word_dict


PHONEME_DICT, WORD_DICT = get_phoneme_word_dict()
# print(PHONEME_DICT)
PHONEME_DICT['SI']=PHONEME_DICT['SIL']
class CMU_MOSI():

	def __init__(self):
		self.base_path = '/home/hyd/workhorse2/multi-hop-attention/code-clean/transformer/Raw/prepro_data/'

	def train(self):
		hubert = np.load(self.base_path + 'X_hubert_train.npy', allow_pickle=True)
		bert = np.load(self.base_path + 'X_bert_2_train.npy', allow_pickle=True)
		emo = np.load(self.base_path + 'Y_emo_train.npy', allow_pickle=True)
		bert_extended = np.load(self.base_path + 'X_bert_extended_train.npy', allow_pickle=True)
		phone = np.load(self.base_path + 'Y_phone_train.npy', allow_pickle=True)
		# trainX, trainY, trainTY, trainVAD, trainPhone, trainBert, trainBertExtend, trainorder
		return hubert, emo, None, None, phone, bert, bert_extended
	def test(self):
		hubert = np.load(self.base_path + 'X_hubert_test.npy', allow_pickle=True)
		bert = np.load(self.base_path + 'X_bert_2_test.npy', allow_pickle=True)
		emo = np.load(self.base_path + 'Y_emo_test.npy', allow_pickle=True)
		bert_extended = np.load(self.base_path + 'X_bert_extended_test.npy', allow_pickle=True)
		phone = np.load(self.base_path + 'Y_phone_test.npy', allow_pickle=True)

		return hubert, emo, None, None, phone, bert, bert_extended

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
	X, Y, Y_trans, Y_vad, Y_phone, X_bert, X_bert_extended, order_files = [], [], [], [], [] , [], [], []
	j = 0
	for i in idx:
		filename='/home/hyd/workhorse2/multi-hop-attention/code-clean/'+'order_'+str(i)+'.txt'
		order_file = list(map(lambda x: x.split(' ')[1].replace('\n', ''), open(filename, 'r').readlines()))
		
		temp_set_X = np.load(os.path.join(path, "X_hubert"+str(i)+".npy"), allow_pickle=True)
		temp_set_Y = np.load(os.path.join(path, "Y_emo"+str(i)+".npy"), allow_pickle=True)
		temp_set_Y_trans = np.load(os.path.join(path, "Y_trans"+str(i)+".npy"), allow_pickle=True) # using ctm trans 
		temp_set_Y_vad = np.load(os.path.join(path, "Y_vad"+str(i)+".npy"), allow_pickle=True)
		temp_set_Y_phone = np.load(os.path.join(path, "Y_phone"+str(i)+".npy"), allow_pickle=True)
		temp_set_X_bert = np.load(os.path.join(path, "X_bert2_"+str(i)+".npy"), allow_pickle=True)
		temp_set_X_bert_extended = np.load(os.path.join(path, "X_bert_extended"+str(i)+".npy"), allow_pickle=True)

		X.append(temp_set_X)
		Y.append(temp_set_Y)
		Y_trans.append(temp_set_Y_trans)
		Y_vad.append(temp_set_Y_vad)
		Y_phone.append(temp_set_Y_phone)
		X_bert.append(temp_set_X_bert)
		X_bert_extended.append(temp_set_X_bert_extended)
		order_files.extend(order_file)

	X = np.concatenate(X)#[:num_ins]
	Y = np.concatenate(Y)#[:num_ins]
	Y_trans = np.concatenate(Y_trans)
	# logging.info(Y_trans)
	# exit()
	Y_vad = np.concatenate(Y_vad)
	Y_phone = np.concatenate(Y_phone)
	X_bert = np.concatenate(X_bert)
	X_bert_extended = np.concatenate(X_bert_extended)
	return (X, Y, Y_trans, Y_vad, Y_phone, X_bert, X_bert_extended, order_files)

def collate_lines(seq_list):
	seq_list = (j for j in seq_list if j is not None)
	# hubert, emo, None, None, phone, bert, bert_extended
	inputs, targets, trans, vad, phone_target, X_bert, X_bert_extend = zip(*seq_list)
	input_lens = [len(seq) for seq in inputs] #input_lens of each in input
	seq_order = sorted(range(len(input_lens)), key=input_lens.__getitem__, reverse=True) # Order of the sorting to sort others
	
	### ordering based on the seq_order ####################
	inputs = [inputs[i] for i in seq_order] #sort the inputs
	targets = [targets[i] for i in seq_order] #sort the targets
	trans = [trans[i] for i in seq_order]# if trans[0] else None  #sort the trans .split(' ') 
	vad = [vad[i] for i in seq_order] if vad[0] else None 
	phone_target =  [phone_target[i] for i in seq_order]
	input_len_sorted = [input_lens[i] for i in seq_order]
	X_bert = [X_bert[i] for i in seq_order]
	X_bert_extend = [X_bert_extend[i] for i in seq_order]
	########################################################

	inputs = [torch.from_numpy(f).cuda() for f in inputs] if cuda\
						else [torch.from_numpy(f) for f in inputs]

	X_bert = [torch.from_numpy(f).cuda() for f in X_bert] if cuda\
						else [torch.from_numpy(f) for f in X_bert]
	
	X_bert_extend = [torch.from_numpy(f).cuda() for f in X_bert_extend] if cuda\
						else [torch.from_numpy(f) for f in X_bert_extend]
	####### getting embedding for each word in the sentence for every batch ########
	####### This has gotton ugly ###################################################
	# embeds = [bert_embedding(tr) for tr in trans]
	# # embeds = list(map(lambda x: x[1], embeds))
	# embeds_only = []
	# for embed_batch in embeds:
	# 	embed_batch_new = []
	# 	for em in embed_batch:
	# 		embed_batch_new.append(em[1][0])
	# 	embeds_only.append(embed_batch_new)

	# embeds_only = list(map(lambda x: torch.Tensor(x), embeds_only))
	# if cuda: embeds_only = [f.cuda() for f in embeds_only]
	# trans_len = [f.shape[0] for f in embeds_only]
	# trans_pad=torch.nn.utils.rnn.pad_sequence(embeds_only, batch_first=True)
	################################################################################

	# embeds = [ em[1][0] for embed_batch in embeds for em in embed_batch ]
	# import pdb 
	# pdb.set_trace()
	

	# phone_target_sequence = list(unique_everseen(phone_target))
	X_bert_extend_len = list(map(lambda x: x.shape[0], X_bert_extend))
	X_bert_extendpad = torch.nn.utils.rnn.pad_sequence(X_bert_extend, batch_first=True)

	X_bert_len = list(map(lambda x: x.shape[0], X_bert))
	X_bertpad = torch.nn.utils.rnn.pad_sequence(X_bert, batch_first=True)

	phone_target_sequence = [list(unique_everseen(ph)) for ph in phone_target]
	# mistake in cmu-mosi preprocess lead all sIL to SI 
	phone_target_sequence_ids = list(map(lambda x: torch.Tensor(list(PHONEME_DICT[ph] for ph in x)), phone_target_sequence))
	phone_target_sequence_ids_len = list(map(lambda x: x.shape[0], phone_target_sequence_ids))

	phone_target_sequence_ids_pad=torch.nn.utils.rnn.pad_sequence(phone_target_sequence_ids, batch_first=True).cuda().unsqueeze(2)
	# import pdb 
	# pdb.set_trace()
	# phone_target_sequence = list(dict.fromkeys(phone_target))
	inputs_pad = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
	phone_target_ids = list(map(lambda x: torch.Tensor(list(PHONEME_DICT[ph] for ph in x)), phone_target))
	phone_target_ids_len = list(map(lambda x: x.shape[0], phone_target_ids))

	phone_target_id_pad = torch.nn.utils.rnn.pad_sequence(phone_target_ids, batch_first=True).cuda().unsqueeze(2)
	
	# if inputs_pad.shape[1]!=phone_target_id_pad.shape[1]:
	# 	# most likely phone_target_id_pad is less in shape 
	# 	# import pdb 
	# 	# pdb.set_trace()
	# 	if phone_target_id_pad.shape[1] > inputs_pad.shape[1]:
	# 		diff = phone_target_id_pad.shape[1] - inputs_pad.shape[1]
	# 		phone_target_id_pad = phone_target_id_pad[:,:-diff,:]
	# 	if phone_target_id_pad.shape[1] < inputs_pad.shape[1]:
	# 		diff = inputs_pad.shape[1] - phone_target_id_pad.shape[1]
	# 		phone_target_id_pad = F.pad(input=phone_target_id_pad, pad=(0, 0, 0, diff), mode='constant', value=0)

	# print(input_len_sorted)
	# print(inputs_pad.shape)
	# print(phone_target_id_pad.shape)
	# import pdb 
	# pdb.set_trace()
	# print(targets)
	targets = torch.LongTensor(targets).cuda()
	input_len_sorted = torch.Tensor(input_len_sorted).cuda()
	X_bert_len = torch.Tensor(X_bert_len).cuda()
	X_bert_extend_len = torch.Tensor(X_bert_extend_len).cuda()
	phone_target_ids_len = torch.Tensor(phone_target_ids_len).cuda()
	phone_target_sequence_ids_len = torch.Tensor(phone_target_sequence_ids_len).cuda()
		
	return inputs_pad, targets, None, vad, X_bertpad, X_bert_len, phone_target_id_pad, \
				input_len_sorted, phone_target_sequence_ids_pad, X_bert_extendpad,  X_bert_extend_len, phone_target_ids_len , phone_target_sequence_ids_len# first None is trans_pad


def collate_lines_context(seq_list):

	# seq_list_tuples = (j for j in seq_list if j is not None)
	items = []
	# import pdb
	# pdb.set_trace()
	for one_seq_idx in range(3):
		seq_list_temp = list(map(lambda x: x[:][one_seq_idx], seq_list))# bfritem, aftitem, curritem
		# print(seq_list)
		inputs, targets, trans, vad, phone_target, X_bert, X_bert_extend = zip(*seq_list_temp)
		input_lens = [len(seq) for seq in inputs] #input_lens of each in input
		seq_order = sorted(range(len(input_lens)), key=input_lens.__getitem__, reverse=True) # Order of the sorting to sort others
		
		### ordering based on the seq_order ####################
		inputs = [inputs[i] for i in seq_order] #sort the inputs
		targets = [targets[i] for i in seq_order] #sort the targets
		trans = [trans[i].split(' ') for i in seq_order] #sort the trans
		vad = [vad[i] for i in seq_order]
		phone_target =  [phone_target[i] for i in seq_order]
		input_len_sorted = [input_lens[i] for i in seq_order]
		X_bert = [X_bert[i] for i in seq_order]
		X_bert_extend = [X_bert_extend[i] for i in seq_order]
		########################################################

		inputs = [torch.from_numpy(f).cuda() for f in inputs] if cuda\
							else [torch.from_numpy(f) for f in inputs]

		X_bert = [torch.from_numpy(f).cuda() for f in X_bert] if cuda\
							else [torch.from_numpy(f) for f in X_bert]
		
		X_bert_extend = [torch.from_numpy(f).cuda() for f in X_bert_extend] if cuda\
							else [torch.from_numpy(f) for f in X_bert_extend]


		X_bert_extend_len = list(map(lambda x: x.shape[0], X_bert_extend))
		X_bert_extendpad = torch.nn.utils.rnn.pad_sequence(X_bert_extend, batch_first=True)

		X_bert_len = list(map(lambda x: x.shape[0], X_bert))
		X_bertpad = torch.nn.utils.rnn.pad_sequence(X_bert, batch_first=True)

		phone_target_sequence = [list(unique_everseen(ph)) for ph in phone_target]
		phone_target_sequence_ids = list(map(lambda x: torch.Tensor(list(PHONEME_DICT[ph] for ph in x)), phone_target_sequence))
		phone_target_sequence_ids_len = list(map(lambda x: x.shape[0], phone_target_sequence_ids))

		phone_target_sequence_ids_pad=torch.nn.utils.rnn.pad_sequence(phone_target_sequence_ids, batch_first=True).cuda().unsqueeze(2)

		inputs_pad = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
		phone_target_ids = list(map(lambda x: torch.Tensor(list(PHONEME_DICT[ph] for ph in x)), phone_target))
		phone_target_ids_len = list(map(lambda x: x.shape[0], phone_target_ids))

		phone_target_id_pad = torch.nn.utils.rnn.pad_sequence(phone_target_ids, batch_first=True).cuda().unsqueeze(2)

		targets = torch.LongTensor(targets).cuda()
		input_len_sorted = torch.Tensor(input_len_sorted).cuda()
		X_bert_len = torch.Tensor(X_bert_len).cuda()
		X_bert_extend_len = torch.Tensor(X_bert_extend_len).cuda()
		phone_target_ids_len = torch.Tensor(phone_target_ids_len).cuda()
		phone_target_sequence_ids_len = torch.Tensor(phone_target_sequence_ids_len).cuda()
		items.append((inputs_pad, targets, None, vad, X_bertpad, X_bert_len, phone_target_id_pad, \
				input_len_sorted, phone_target_sequence_ids_pad, X_bert_extendpad,  X_bert_extend_len, phone_target_ids_len , phone_target_sequence_ids_len))
	return items


class Dataset(Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, trainX, trainY, trainTY, trainVAD, trainPhone, trainBert, trainBertExtend, dataset = 'IEMOCAP', test=False):
		'Initialization'
		self.labels = trainY
		self.labelT = trainTY
		self.X = trainX
		self.test = test
		self.labelVAD = trainVAD
		self.labelPhone = trainPhone
		self.Xbert = trainBert
		self.XBertExtend = trainBertExtend
		self.dataset = dataset
		self.phoneme_mapping = read_phonemes()
		# print(self.phoneme_mapping)

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.X)

	def binlabel(self, label):
		if label > 0:
			return 1
		else: return 0
		 # strongly positive (labeled as +3), positive (+2), weakly positive (+1), neutral (0), weakly negative (–1), negative (–2), strongly negative (–3), and uncertain

	def get_feat_by_words(self, mfc, label, trans_label, vad, phone_seg, xbert, xbert_extend, NUM_OF_WORDS):
		# print( mfc, label, trans_label, vad, phone_seg, xbert, xbert_extend, NUM_OF_WORDS)
		# print(trans_label)
		trans_label_count = []
		prev, count = trans_label[0], 1
		for curr in trans_label[1:]:
			if curr == prev:
				count += 1 
			else:
				trans_label_count.append((prev, count))
				count=1
				prev=curr
		trans_label_count.append((prev,count))
		# print(trans_label_count)
		
		if NUM_OF_WORDS == 1:
			if trans_label_count[0][0]=='s':
				start, end = trans_label_count[0][1], trans_label_count[1][1]+trans_label_count[0][1]
			else:
				start, end = 0, trans_label_count[0][1]

		else:
			local_num_of_words = min(NUM_OF_WORDS, len(trans_label_count))
			# print(local_num_of_words)
			start, end=0, 0 
			for j in range(local_num_of_words):
				end+=trans_label_count[j][1]
			# end=end+trans_label_count[local_num_of_words][1]
		import pdb
		# pdb.set_trace()
		# print(start, end)
		mfc, label, trans_label, vad, phone_seq, xbert, xbert_extend = \
			mfc[start:end], label, trans_label[start:end], vad, phone_seg[start:end], xbert, xbert_extend[start:end]
		return mfc, label, trans_label, vad, phone_seq, xbert, xbert_extend

	def __getitem__(self, index):
		# print(new_word2idx)
		'Generates one sample of data'
		if self.dataset == 'IEMOCAP':
			label = emotions_used.index(self.labels[index])
		if self.dataset =='CMU_MOSI':
			label = self.binlabel(self.labels[index])
		mfc = self.X[index]

		trans_label = self.labelT[index] if self.labelT!=None else None 
		# print(self.labelT[index])
		trans_label = self.labelT[index].lstrip().split(' ') if self.labelT!=None else None 
		if trans_label and len(trans_label) < 2:
			trans_label = None 

		trans_label = np.array(trans_label)

		
		# mfc, phone_seg = clean_mfc(self.X[index],self.labelPhone[index],self.phoneme_mapping ) # removes the frames with silence phoneme 
		phone_seg = np.array(self.labelPhone[index])
		xbert = self.Xbert[index]
		xbert_extend = self.XBertExtend[index]
		vad = self.labelVAD[index] #if self.labelVAD!=None else None 

		if mfc.shape[0] == 0:
			# print("Returning None")
			return None
		

		# print(mfc.shape, xbert_extend.shape, phone_seg.shape )
		idx = np.arange(1,xbert_extend.shape[0],2)
		xbert_extend = xbert_extend[idx]
		idx2 = np.arange(1,phone_seg.shape[0],2)
		phone_seg = np.array(phone_seg)[idx2]
		idx3 = np.arange(1,len(trans_label),2)
		trans_label = trans_label[idx3]

		# print(mfc.shape, xbert_extend.shape, phone_seg.shape )
		if xbert_extend.shape[0] !=  mfc.shape[0]:
			if xbert_extend.shape[0] >  mfc.shape[0]:
				diff = xbert_extend.shape[0] -  mfc.shape[0]
				xbert_extend = xbert_extend[:-diff,:]
			if xbert_extend.shape[0] <  mfc.shape[0]:
				diff =  mfc.shape[0] - xbert_extend.shape[0] 
				xbert_extend = np.pad(xbert_extend, ((diff,0),(0,0)),mode='constant', constant_values=0 )

		if phone_seg.shape[0] !=  mfc.shape[0]:
			if phone_seg.shape[0] >  mfc.shape[0]:
				diff = phone_seg.shape[0] -  mfc.shape[0]
				phone_seg = phone_seg[:-diff]
			if phone_seg.shape[0] <  mfc.shape[0]:
				diff =  mfc.shape[0] - phone_seg.shape[0] 
				phone_seg = np.pad(phone_seg, ((diff,0)),mode='constant', constant_values='SIL' )

		## making trans_labels the same shape as mfc 
		if trans_label.shape[0] !=  mfc.shape[0]:
			if trans_label.shape[0] >  mfc.shape[0]:
				diff = trans_label.shape[0] -  mfc.shape[0]
				trans_label = trans_label[:-diff]
			if trans_label.shape[0] <  mfc.shape[0]:
				diff =  mfc.shape[0] - trans_label.shape[0] 
				trans_label = np.pad(trans_label, ((diff,0)),mode='constant', constant_values='sil' )


		# print(mfc.shape, xbert_extend.shape, phone_seg.shape )
		# import pdb 
		# pdb.set_trace()
		# print(trans_label)
		# print(trans_label)
		mfc, label, trans_label, vad, phone_seg, xbert, xbert_extend = \
			self.get_feat_by_words(mfc, label, trans_label, vad, phone_seg, xbert, xbert_extend, NUM_OF_WORDS)

		# print(trans_label)
		return mfc, label, trans_label, vad, phone_seg, xbert, xbert_extend


class DatasetContext(Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, trainX, trainY, trainTY, trainVAD, trainPhone, trainBert, trainBertExtend, orderfiles, dataset = 'IEMOCAP',test=False):
		'Initialization'
		self.labels = trainY
		self.labelT = trainTY
		self.X = trainX
		self.test = test
		self.labelVAD = trainVAD
		self.labelPhone = trainPhone
		self.Xbert = trainBert
		self.XBertExtend = trainBertExtend
		self.phoneme_mapping = read_phonemes()
		self.orderfiles = orderfiles 
		self.dialogorder = self.getdialogeorder()
		# print(self.labels)

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.X)

	def getdialogeorder(self,):
		def getkey(l):
			return l.split('/')[-1].split('.')[0]

		dialogorder={}
		files = list(map(lambda x: '/home/hyd/workhorse2/multi-hop-attention/code-clean/transformer/dialoge_order/'+x, os.listdir('/home/hyd/workhorse2/multi-hop-attention/code-clean/transformer/dialoge_order/')))
		for f in files:
			with(open(f, 'r')) as fopen:
				lines = fopen.readlines()
				lines = list(map(lambda x: x.replace('\n', ''), lines))

			for i, l in enumerate(lines):
				key = getkey(l)
				if i == 0:
					before = None 
					after = getkey(lines[1])
				elif i == len(lines)-1:
					before=prev_key
					after=None
				else:
					before=prev_key
					after=getkey(lines[i+1])
				
				if before in self.orderfiles: dialogorder[key]=[before]
				else: dialogorder[key]=[None]
				if after in self.orderfiles: dialogorder[key].append(after)
				else: dialogorder[key].append(None)
				
				prev_key=key
				# if i == len(lines)-1: 
					# after=None 
				# after=getkey(lines[i+2])

		return dialogorder

	def __getitemone__(self, index):
		################################## Features for the current index ############################################
		# print(new_word2idx)
		'Generates one sample of data'
		# print(index, self.labels[index])
		label = emotions_used.index(self.labels[index])
		trans_label = self.labelT[index]
		# print(self.labelT[index])
		trans = self.labelT[index].lstrip().split(' ')
		mfc = self.X[index]
		
		if len(trans) < 2:
			trans_label = "None"
		# mfc, phone_seg = clean_mfc(self.X[index],self.labelPhone[index],self.phoneme_mapping ) # removes the frames with silence phoneme 
		phone_seg = self.labelPhone[index]
		xbert = self.Xbert[index]
		xbert_extend = self.XBertExtend[index]

		if mfc.shape[0] == 0:
			# print("Returning None")
			return None
		# import pdb 
		# pdb.set_trace()
		idx = np.arange(1,xbert_extend.shape[0],2)
		xbert_extend = xbert_extend[idx]
		phone_seg = np.array(phone_seg)[idx]

		if xbert_extend.shape[0] !=  mfc.shape[0]:
			if xbert_extend.shape[0] >  mfc.shape[0]:
				diff = xbert_extend.shape[0] -  mfc.shape[0]
				xbert_extend = xbert_extend[:-diff,:]
			if xbert_extend.shape[0] <  mfc.shape[0]:
				diff =  mfc.shape[0] - xbert_extend.shape[0] 
				xbert_extend = np.pad(xbert_extend, ((diff,0),(0,0)),mode='constant', constant_values=0 )

		if phone_seg.shape[0] !=  mfc.shape[0]:
			if phone_seg.shape[0] >  mfc.shape[0]:
				diff = phone_seg.shape[0] -  mfc.shape[0]
				phone_seg = phone_seg[:-diff,:]
			if phone_seg.shape[0] <  mfc.shape[0]:
				diff =  mfc.shape[0] - phone_seg.shape[0] 
				phone_seg = np.pad(phone_seg, ((diff,0)),mode='constant', constant_values='SIL' )
		######################################################################################################



		return mfc, label, trans_label, self.labelVAD[index], phone_seg, xbert, xbert_extend
	
	def __getitem__(self, index):
		
		filename=self.orderfiles[index]
		beforefile, afterfile = self.dialogorder[filename]
		bfrindx = self.orderfiles.index(beforefile) if beforefile else None 
		aftindx = self.orderfiles.index(afterfile) if afterfile else None 
		#######3 For combining in the input space #############
		if bfrindx:
			mfc1, label1, trans_label1, vad1, phone_seg1, xbert1, xbert_extend1 = self.__getitemone__(bfrindx)

		if aftindx:
			mfc3, label3, trans_label3, vad3, phone_seg3, xbert3, xbert_extend3 = self.__getitemone__(aftindx)

		mfc2, label2, trans_label2, vad2, phone_seg2, xbert2, xbert_extend2 = self.__getitemone__(index)
	
		if not bfrindx:
			mfc1, label1, trans_label1, vad1, phone_seg1, xbert1, xbert_extend1 = \
				mfc2, label2, trans_label2, vad2, phone_seg2, xbert2, xbert_extend2
		if not aftindx:
			mfc3, label3, trans_label3, vad3, phone_seg3, xbert3, xbert_extend3 = \
				mfc2, label2, trans_label2, vad2, phone_seg2, xbert2, xbert_extend2

		mfc = np.concatenate((mfc1, mfc2, mfc3), axis=0)
		phone_seg = np.concatenate((phone_seg1, phone_seg2, phone_seg3), axis=0)
		xbert = np.concatenate((xbert1, xbert2, xbert3), axis=0)
		xbert_extend = np.concatenate((xbert_extend1, xbert_extend2, xbert_extend3), axis=0)
		res = (mfc, label2, trans_label2, vad2, phone_seg, xbert, xbert_extend)
		#######3 For combining in the input space #############

		curritem = self.__getitemone__(index)
		if bfrindx:
			bfritem= self.__getitemone__(bfrindx)
		else:
			bfritem = curritem 
		if aftindx:
			aftitem= self.__getitemone__(aftindx)
		else:
			aftitem = curritem
		return bfritem, aftitem, curritem



def plot_attention(attention_weights, name):
	# print(attention_weights.shape)
	fig = plt.figure()
	plt.imshow(attention_weights)
	fig.savefig(name)
	plt.close()
	return 

def output_mask(maxlen, lengths):
	lens = lengths.unsqueeze(0)
	ran = torch.arange(0, maxlen, 1, out=lengths.new()).unsqueeze(1)
	mask = (ran < lens).float()
	return mask

def calculate_attention2(keys, query, context_lengths, eps=1e-9):
	query = query.unsqueeze(2)
	attn = torch.bmm(keys, query)
	mask = output_mask(max(context_lengths), context_lengths).transpose(0, 1).cuda() if cuda\
			else output_mask(max(context_lengths), context_lengths).transpose(0,1)
	attn = attn.view(mask.shape[0], mask.shape[1], -1)
	attn = attn * mask.unsqueeze(2)
	emax = torch.max(attn, 1)[0].unsqueeze(1)
	eval = torch.exp(attn - emax)
	attn = eval / (eps + eval.sum(1).unsqueeze(1))
	attn = attn.view(attn.shape[0], -1)
	return attn.squeeze()

def calculate_context2(attn, context):
	# attn = B, TxF
	# context  = B, TxF, C
	attn = attn.unsqueeze(1)
	ctx = torch.bmm(attn, context).squeeze()
	return ctx

class self_attn_model(nn.Module):
	def __init__(self, channel_0, channel_1, channel_2, featureD, class_n, target_vocab):
		super(self_attn_model, self).__init__()
		self.target_vocab = target_vocab
		self.featureD = featureD
		self.convlayers = nn.Sequential(
			nn.Conv2d(channel_0, channel_1, 3, 1, bias=False, padding=True),
			nn.ELU(inplace=True),
			nn.Conv2d(channel_1, channel_2, 3, 1, bias=False, padding=True),
			nn.ELU(inplace=True),
			nn.Conv2d(channel_2, self.featureD, 3, 1, bias=False, padding=True),
		)
		self.textBRE_lstm = nn.LSTM(input_size = 768, hidden_size=featureD, \
								num_layers=1, bidirectional=True)
		self.bn = nn.BatchNorm1d(self.featureD, affine=False)
		self.fc1 = nn.Linear(self.featureD, class_n)
		self.fc2 = nn.Linear(self.featureD, self.featureD)
		self.fc_vad = nn.Linear(class_n, 3)
		
		self.query_proj = nn.Linear(2*featureD, featureD)
 
	def forward(self, x1, x2):
		features_x1 = [torch.from_numpy(f).cuda() for f in x1] if cuda\
						else [torch.from_numpy(f) for f in x1]
		lens_x1 = torch.Tensor([f.shape[0] for f in x1]) # audio lens 
		## text 
		out = rnn.pad_sequence(x2)
		out = out.cuda() if cuda else out
		_, (text_hidden, _) = self.textBRE_lstm(out)
		_, b, _ = text_hidden.size()
		text_hidden = text_hidden.transpose(0,1).reshape(b, -1)
		out = rnn.pad_sequence(features_x1, batch_first=True)
		b, t, f = out.shape[0], out.shape[1], out.shape[2]
		out = out.unsqueeze(1)
		out = self.convlayers(out)
		c = out.shape[1]
		out = out.permute(0, 2, 3, 1) # B, T, F, C
		out = out.view(b, -1, c) # B, TxF, C
		out = self.fc2(out)
		query_in = self.query_proj(text_hidden)
		attn = calculate_attention2(out, query_in, lens_x1) #B, TxF
		# plot_attention(attn[0].view(t, f).data.cpu().numpy(), \
			# phoneme_class+"/attention_weights.png")
		feat = calculate_context2(attn, out) # B, C
		out = self.fc1(feat) # C, class_n
		out_vad = self.fc_vad(out) # C, 3
		out = nn.functional.log_softmax(out, dim=1)
		return out, out_vad, feat, attn.view(b, t, f)

class Net(nn.Module):
	def __init__(self, channel_0, channel_1, channel_2, channel_3, channel_4, class_n):
		super(Net, self).__init__()
		kernel_size = 2
		stride = 2
		self.featureD = 64
		self.convlayers = nn.Sequential(
			nn.Conv2d(channel_0, channel_1, (kernel_size, kernel_size), (stride, 2), bias=False),
			nn.BatchNorm2d(channel_1),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel_1, channel_2, (kernel_size, kernel_size), (stride, 2), bias=False),
			nn.BatchNorm2d(channel_2),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel_2, channel_3, (kernel_size, kernel_size), (stride, 2), bias=False),
			nn.BatchNorm2d(channel_3),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel_3, channel_4, (kernel_size, kernel_size), (stride, 2), bias=False),
			nn.BatchNorm2d(channel_4),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel_4, self.featureD, (kernel_size, kernel_size), (stride, 2), bias=False),
		)
		self.bn = nn.BatchNorm1d(self.featureD, affine=False)

		self.fc1 = nn.utils.weight_norm(nn.Linear(self.featureD, class_n, bias=False), name='weight')

	def forward(self, x):
		# print("1: ", x[0].shape)
		x = [torch.Tensor(f).cuda() for f in x] if cuda\
						else [torch.Tensor(f) for f in x]
		x = rnn.pad_sequence(x, batch_first=True).unsqueeze(1)
		# print("2: ", x.shape, x[0].shape)
		b, c, t, f = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

		x = self.convlayers(x)
		# print("3: ", x.shape)
		x = F.avg_pool2d(x, [x.size()[2], x.size()[3]], stride=1)
		# print("4: ", x.shape)
		x = x.view(-1, self.featureD)
		# print("5: ", x.shape)
		x = self.bn(x)
		return F.log_softmax(self.fc1(x), dim=-1) # x, self.fc1(x)

if __name__ == "__main__":

	# spkd = {}
	# for spkid in range(10):
	# 	labelcount={}
	# 	(X, Y, Y_trans, Y_vad, Y_phone, X_bert, X_bert_extended, order_files) = load_raw(os.environ['DATA_PATH'],[spkid])
	# 	for emo in Y:
	# 		if emo in labelcount: labelcount[emo]+=1
	# 		else: labelcount[emo]=1
	# 	spkd[spkid] = labelcount

	# for emo in emotions_used:
	# 	spke = ''
	# 	for spkid in range(10):

	# 		if spkid < 10: spke += str( spkd[spkid][emo]) + ' & '
	# 		else: spke += str(spkd[spkid][emo]) 
	# 	print( emo + " & " + spke)
	# exit()
	# read_phonemes()

	loader =  CMU_MOSI()
	trainX, trainY, trainTY, trainVAD, trainPhone, trainBert, trainBertExtend = loader.train()
	devX, devY, devTY, devVAD, devPhone, devBert, devBertExtend = loader.test()
	import pdb 
	pdb.set_trace()

	exit()
	train_idx=[1,3]
	test_idx=[2]
	dev_idx=[2]
	loader = IEMOCAP(train_idx, test_idx, dev_idx)
	trainX, trainY, trainTY, trainVAD, trainPhone, trainBert, trainBertExtend, trainorder = loader.train()
	trainTY = list(map(lambda sentence: (' ').join(sentence).replace("\n", "").lower().translate(str.maketrans('','',string.punctuation)) , trainTY))
	# print(trainorder)
	train = DatasetContext(trainX, trainY, trainTY, trainVAD, trainPhone, trainBert, trainBertExtend, trainorder)
	# dataloader_args = dict(shuffle=True, batch_size=BATCH_SIZE, collate_fn = collate_lines_context, num_workers=0) 

	# train_loader = dataloader.DataLoader(train, **dataloader_args)
	seq_list_tuples = []
	for i in range(4):
		seq_list_tuples.append(train[i])
	collate_lines_context(seq_list_tuples)