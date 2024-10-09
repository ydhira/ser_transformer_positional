import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn
import bcolz, pickle
import numpy as np 
# from pytorch_pretrained_bert import BertTokenizer, BertModel
from bert_embedding import BertEmbedding
import sys
import re
import matplotlib.pyplot as plt

# ctc_path = "/home/hyd/DL-Spring19/Hw3/hw3p2/code/"
# sys.path.insert(1, ctc_path)
# from myCTC import *

# glove_path = "/home/hyd/multi-hop-attention/data"
# vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
# words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
# word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))
# global new_word2idx
# new_word2idx = {'PAD': 0}
# print("hello changing from vs code")
emotions_used = ['ang', 'hap','sad', 'neu'] # previosuly used emotion classes 
# emotions_used = ['ang', 'hap', 'fru', 'sur', 'fea', 'exc', 'dis', 'sad', 'neu']
cuda = True 

def  clean_mfc(mfc, phone_seg ) :
	if phone_seg is None:
		print("NONEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
		return mfc 
	phone_seg.reverse()
	new_phone_seg = []
	for ps in phone_seg:
		if ps[0] == "'SIL'":
			mfc = np.delete(mfc, np.s_[ps[1]::ps[2]], 0)
		else:
			new_phone_seg.append(ps)
	new_phone_seg.reverse()
	return mfc , new_phone_seg


class Dataset(Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, trainX, trainY, trainTY, trainVAD, trainPhone, test=False):
		'Initialization'
		self.labels = trainY
		self.labelT = trainTY
		self.X = trainX
		self.test = test
		self.labelVAD = trainVAD
		self.labelPhone = trainPhone

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.X)

	def __getitem__(self, index):
		# print(new_word2idx)
		'Generates one sample of data'
		label = emotions_used.index(self.labels[index])
		trans = []
		# print(self.labelT[index].split(' '))
		##########3 if using the glov embeds #############33
		# if self.test:
		# 	for i in self.labelT[index].split(' '):
		# 		# try:
		# 		# print(i)

		# 		try:
		# 			trans.append(new_word2idx[i])
		# 		except KeyError:
		# 			trans.append(0)
		# else:
		# 	trans = [ new_word2idx[i] for i in self.labelT[index].split(' ') ]
		##########3 if using the glov embeds #############33
		trans_label = self.labelT[index]
		trans = self.labelT[index].lstrip().split(' ')
		if len(trans) < 2:
			trans_label = "None"

		# ignore the words that doent occure in the new_word2idx
		# trans = torch.LongTensor(trans)
		# print(trans)
		#return self.X[index], label, trans
		mfc, new_phone_seg = clean_mfc(self.X[index],self.labelPhone[index] ) # removes the frames with silence phoneme 
		# print(self.X[index].shape, len(self.labelPhone[index] ))
		# print(mfc.shape, len(new_phone_seg))
		# print(new_phone_seg)
		# exit()
		return mfc, label, trans_label, self.labelVAD[index], new_phone_seg #self.labelPhone[index]

def plot_attention(attention_weights, name):
	# print(attention_weights.shape)
	fig = plt.figure()
	plt.imshow(attention_weights)
	fig.savefig(name)
	plt.close()
	return 

def init_embeding(target_vocab):
	#######################################################################
	new_word2idx = {}
	glove = {w: vectors[word2idx[w]] for w in words}
	emb_dim = 50
	#######################################################################
	matrix_len = len(target_vocab) + 1
	weights_matrix = np.zeros((matrix_len , 50))
	words_found = 0
	# print('initializing matrix ', matrix_len, len(word2idx.keys()), len(new_word2idx.keys()))

	for i, word in enumerate(target_vocab):
		i = i + 1 # 0 is pad
		new_word2idx[word] = i
		try: 
			weights_matrix[i] = glove[word]
			words_found += 1
		except KeyError:
			# print("not in the dict :",word, i)
			weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
	return weights_matrix, new_word2idx

def create_emb_layer(weights_matrix, non_trainable=False):
	# print(weights_matrix)
	weights_matrix = torch.Tensor(weights_matrix)
	num_embeddings, embedding_dim = weights_matrix.shape
	# print("--->", num_embeddings, embedding_dim)
	emb_layer = nn.Embedding(num_embeddings, embedding_dim)
	emb_layer.load_state_dict({'weight': weights_matrix})
	if non_trainable:
		emb_layer.weight.requires_grad = False

	return emb_layer, num_embeddings, embedding_dim

def output_mask(maxlen, lengths):
	# print(maxlen)
	lens = lengths.unsqueeze(0)
	ran = torch.arange(0, maxlen, 1, out=lengths.new()).unsqueeze(1)
	mask = (ran < lens).float()
	return mask

def calculate_attention(keys, context, context_lengths, orig_shape, eps=1e-9):
	print("\t\t Calculating Attention ")
	# if context_lengths:
		# context_lengths =  torch.Tensor(context_lengths)
		# print(context_lengths, max(context_lengths))
	L = max(context_lengths)# - 6
	mask = output_mask(L, context_lengths).transpose(0,1).cuda() if cuda\
		else output_mask(L, context_lengths).transpose(0,1)

	# keys = keys.unsqueeze(1)
	context = context.unsqueeze(2)
	print(keys.shape, context.shape)
	energy = torch.bmm(keys, context).squeeze() 
	energy = energy.view(-1, orig_shape[2], orig_shape[1])
	print("\t1: ", energy.shape)
	print("\t2: ", mask.unsqueeze(1).shape)
	# if context_lengths:
	energy = energy * mask.unsqueeze(1)  # (BS, L) 
	emax = torch.max(energy, 1)[0].unsqueeze(1)
	print("\t3: ", emax, emax.shape)
	eval = torch.exp(energy - emax)# * mask
	print("\t4: ", eval, eval.shape)
	attn = eval / (eps + eval.sum(1).unsqueeze(1))  # (BS, L)
	return attn

def calculate_attention2(keys, query, context_lengths, eps=1e-9):
	# print("Inside attention")
	query = query.unsqueeze(2)
	# print(keys.shape, query.shape)
	attn = torch.bmm(keys, query)
	# attn = attn.view(-1, t, f, c)
	# print(attn.shape)
	mask = output_mask(max(context_lengths), context_lengths).transpose(0, 1).cuda() if cuda\
			else output_mask(max(context_lengths), context_lengths).transpose(0,1)
	# mask = mask.reshape(mask.shape[0], -1)
	# print(mask.shape)
	attn = attn.view(mask.shape[0], mask.shape[1], -1)
	# print(attn.shape, mask.unsqueeze(2).shape)
	attn = attn * mask.unsqueeze(2)
	# print("attn: ", attn.shape)
	emax = torch.max(attn, 1)[0].unsqueeze(1)
	# print("emax:", emax.shape)
	eval = torch.exp(attn - emax)
	# print(eval.shape)
	attn = eval / (eps + eval.sum(1).unsqueeze(1))
	# print("---->:", attn.shape)
	attn = attn.view(attn.shape[0], -1)
	return attn.squeeze()

def calculate_context2(attn, context):
	# attn = B, TxF
	# context  = B, TxF, C
	attn = attn.unsqueeze(1)
	ctx = torch.bmm(attn, context).squeeze()
	# print(ctx.shape)
	return ctx

def calculate_context(attn, context):
	"""

	:param attn:  (BS, L)
	:param context: (BS, L, cdim)
	:return:
	"""
	print("\t\t calculating context")
	print(attn.shape, context.shape )
	# attn = attn.unsqueeze(1)
	ctx = torch.bmm(attn, context).squeeze(1)  # (BS, cdim)
	
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
		# lens_x2 = torch.Tensor([f.shape[0] for f in x2]) # trans lens 

		## text 
		out = rnn.pad_sequence(x2)
		out = out.cuda() if cuda else out
		
		_, (text_hidden, _) = self.textBRE_lstm(out)
		_, b, _ = text_hidden.size()
		text_hidden = text_hidden.transpose(0,1).reshape(b, -1)
		###############
		out = rnn.pad_sequence(features_x1, batch_first=True)
		b, t, f = out.shape[0], out.shape[1], out.shape[2]
		# print("B: %d, T: %d, F: %d" %(b, t, f))
		# print("0: ", out.shape) # B, T, F
		out = out.unsqueeze(1)
		# print("1: ", out.shape) # B, 1, T, F
		out = self.convlayers(out)
		# print("2: ", out.shape) # B, C, T, F
		c = out.shape[1]
		out = out.permute(0, 2, 3, 1) # B, T, F, C
		# print("3: ", out.shape) 
		out = out.view(b, -1, c) # B, TxF, C
		# print("4: ", out.shape)  
		out = self.fc2(out)
		### OLD QUERY USED:
			# print("5: ", out.shape) # B, TxF, C
			# out_query = out.permute(0, 2, 1).view(b, c, t, f)
			# print(out_query.shape) # B, C, T, F
			# query_in = F.avg_pool2d(out_query, [out_query.shape[2], out_query.shape[3]]).squeeze()
			# print("6: ", query_in.shape) # B, C
			# query_in = self.query_proj(query_in)
			# print(query_in.shape)
		query_in = self.query_proj(text_hidden)
		# print(query_in.shape)
		attn = calculate_attention2(out, query_in, lens_x1) #B, TxF
		# print(attn.shape)
		# print(attn.view(b, t, f).shape)
		
		# print(attn[0].shape, attn[0].view(t, f).shape )
		plot_attention(attn[0].view(t, f).data.cpu().numpy(), \
            "attention_weights.png")
		# # query = self.query_proj(query_in)
		# # print("4: ", query.shape)
		# out = out.transpose(1,2).transpose(2, 3)
		# print("5: ", out.shape)
		# attn = calculate_attention(out.contiguous()\
		# 								.view(out.shape[0], -1, out.shape[-1]), out, lens_x1, out.shape)
		# print("6: ", attn.shape)
		# # attn = calculate_attention( )
		feat = calculate_context2(attn, out) # B, C
		# print("feat: ", feat.shape)		
		out = self.fc1(feat) # C, class_n
		out_vad = self.fc_vad(out) # C, 3
		# print(out.shape, out_vad.shape, feat.shape)
		out = nn.functional.log_softmax(out, dim=1)
		return out, out_vad, feat, attn.view(b, t, f)


class multi_attention_model(nn.Module):
	def __init__(self, hidden_size, nlayers, class_n, target_vocab):
		# target_vocab.append('PAD')
		# global new_word2idx
		# print(target_vocab)
		embedding_size = 16
		super(multi_attention_model, self).__init__()
		# audioBRE = nn.ModuleList()
		self.audioBRE_lstm = nn.LSTM(input_size = 40, hidden_size=embedding_size, \
								num_layers=nlayers, bidirectional=True)
		self.audioBRE_fc = nn.Linear(2*embedding_size,class_n)
		# self.audioBRE_L = audioBRE

		# weights_matrix, new_word2idx = init_embeding(target_vocab)
		# print(len(new_word2idx.keys()))
		# embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True) 
		# self.embedding = embedding
		# self.embedding = nn.Embedding()

		self.textBRE_lstm = nn.LSTM(input_size = 768, hidden_size=hidden_size, \
								num_layers=nlayers, bidirectional=True)
		self.textBRE_fc =  nn.Linear(2*hidden_size,class_n)

		self.fc = nn.Linear(4*embedding_size,class_n)
		nn.init.xavier_uniform_(self.fc.weight)
		self.fc.bias.data.fill_(0.1)
		# self.audioBRE = nn.Sequential( *audioBRE)
		# self.textBRE = nn.Sequential( *textBRE)
		# self.ctcmodel = CTCPhonemeSegmentor(47,512,3)
		# self.ctcmodel = self.ctcmodel.cuda()
		# checkpoint = torch.load('/home/hyd/for_shahan/14.model')
		# self.ctcmodel.load_state_dict(checkpoint['model_state_dict'])

		self.fc_vad = nn.Linear(class_n, 3) #nn.Sequential(nn.LeakyReLU(), \
					
					#	)
		self.text_layer = nn.Linear(768, embedding_size)
		# nn.init.xavier_uniform_(self.fc_vad.weight)
		# self.fc_vad[-1].bias.data.fill_(0.1)


	def forward(self, x1, x2):
		# print(x2.shape)
		'''
		x1 is the audio representation. 
		x2 is the indices of the words in the text
		'''
		# phone_log_softmax_logits,  _ = self.ctcmodel(x1)
		# phone_log_softmax_logits = phone_log_softmax_logits.transpose(0, 1)
		# print("---", x1.shape)
		features_x1 = [torch.from_numpy(f).cuda() for f in x1] if cuda\
						else [torch.from_numpy(f) for f in x1]
		# print(len(features_x1))
		lens_x1 = torch.Tensor([f.shape[0] for f in x1]) # audio lens 
		lens_x2 = torch.Tensor([f.shape[0] for f in x2]) # trans lens 
		# sort x2 here 
		# print(lens_x2)
		
		# print(phone_log_softmax_logits.shape) # e.g (2435, 32, 47)

		# x2_sorted_lengths, order = torch.sort(lens_x2, 0, descending=True)
		# _, backorder = torch.sort(order, 0)
		
		out = rnn.pad_sequence(features_x1)#.transpose(0, 1)
		# print("0: ", out.shape)
		# exit()
		audio_lstm, (audio_hidden, _) = self.audioBRE_lstm(out) 
		# audio_hidden = audio_hidden[-2:, :, :]
		# audio_hidden is the hidden state of the last time stamp 
		# audio_lstm is the hidden states of all the time stamps 

		# audio_hidden = audio_hidden.view()
		# print("1: ", audio_lstm.shape, audio_hidden.shape)

		_, b, _ = audio_hidden.size()
		audio_hidden = audio_hidden.transpose(0,1).reshape(b, -1)#.contiguous().view(b, -1)
		# print(audio_hidden.shape)
		# print("------")
		# print(audio_lstm[-1,:,:])
		# exit()
		print("audio hidden: ", audio_hidden.shape)
		# print("3: ", audio_out.shape)
		# print(x2)
		out = rnn.pad_sequence(x2)
		out = out.cuda() if cuda else out
		# print("4: ", out.shape)
		# print(out.shape)
		# print(order, type(order), order.data)
		# out = out[:, order]
		# out = self.embedding(out)
		# print("5: ", out.shape)
		# out = x2 # put this here coz using bert embeddings. No need to use embedding layer
				# previously I was using the embedding layer so used lens_x2, order, backorder 
				# uncomment those if needed. 
		text_lstm, (text_hidden, _) = self.textBRE_lstm(out)
		# print(text_hidden.shape)
		text_hidden = self.text_layer(text_hidden)
		
		# exit()
		# text_hidden = x2[0]
		# text_lstm  = np.array(x2[1])
		# lens_x2 = torch.Tensor([f.shape[1] for f in text_lstm])
		# x2_sorted_lengths, order = torch.sort(lens_x2, 0, descending=True)
		# _, backorder = torch.sort(order, 0)
		# text_lstm = rnn.pad_sequence(text_lstm[order])
		# # unsort it here
		# print(text_lstm.shape, text_hidden.shape)
		# text_lstm = text_lstm[:, backorder, :]
		# text_hidden = text_hidden[:, backorder, :]

		# text_hidden = text_hidden[-2:, :, :]
		# print("4: ", text_lstm.transpose(0, 1).transpose(1, 2).shape)
		_, b, _ = text_hidden.size()
		text_hidden = text_hidden.transpose(0,1).reshape(b, -1)
		print("7: ", text_hidden.shape)
		exit()

		# attn = calculate_attention(audio_hidden, text_lstm.transpose(0, 1).transpose(1, 2), lens_x2)
		# # transpose being batch at the 0th index
		# H1 = calculate_context(attn, text_lstm.transpose(0, 1))

		# attn2 = calculate_attention(H1, audio_lstm.transpose(0, 1).transpose(1, 2), lens_x1)
		# H2 = calculate_context(attn2, audio_lstm.transpose(0, 1))
		# starting with audio first 
		#############3
		# Hs = []
		# K, C, lens = text_hidden, audio_lstm, lens_x1
		# #Hs.append(K) # could be removed 
		# for i in range(1):

		# 	attn = calculate_attention(K, C.transpose(0, 1).transpose(1, 2), lens)
		# 	H = calculate_context(attn, C.transpose(0, 1))
		# 	Hs.append(H)
		# 	K = H
		# 	C = audio_lstm if i % 2 == 1 else text_lstm
		# 	lens = lens_x1 if i % 2 == 1 else lens_x2


		# out = torch.cat((Hs[-1], Hs[-2]), dim=1)
		##############
		# print(audio_hidden.shape, text_hidden.shape)
		feature = torch.cat((text_hidden, audio_hidden), dim=1)
		# print(out.shape)
		# exit()
		# print(out.shape, phone_log_softmax_logits.shape)
		# attn = calculate_attention(out, phone_log_softmax_logits, None)
		# print(attn.shape)

		out = self.fc(feature)
		out_vad = self.fc_vad(out)
		out = nn.functional.log_softmax(out, dim=1)
		# out_vad = nn.functional.log_softmax(out_vad, dim=1)
		return out, out_vad, feature


# class TransEmotion(nn.Module):
# 	def __init__(self,  vocab_size, hidden_size, nlayers ,classn): # 47, 128, 1
# 		super(TransEmotion,self).__init__()

# 		self.rnn = nn.LSTM(input_size = 50, hidden_size=hidden_size, num_layers=nlayers, bidirectional=True, dropout=0.1) # 1 layer, batch_size = False
# 		self.fc1 = nn.Linear(2*hidden_size,1024)
# 		self.do = nn.Dropout(p=0.3)
# 		self.relu = nn.ReLU()
# 		self.scoring = nn.Linear(1024,classn)
# 		weights_matrix = init_embeding()
# 		self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True) 

# 	def forward(self, x):
# 		lens =  [len(f) for f in x]
# 		x = rnn.pad_sequence(x).cuda()
# 		x = self.embedding(x)
# 		# print("Embedding shape", x.shape)
		
# 		input_features = x.unsqueeze(1).permute(2,1,3,0) # (bs, 1, 40, max_len) bsx1xmax_lenx40
# 		# print("---->", input_features.size())
# 		n, c, h, w = input_features.size()
# 		input_features = input_features.view(n, c*h, w).permute(2, 0, 1)
# 		packed_input = rnn.pack_padded_sequence(input_features, lens) # packed version
# 		output_packed, hidden = self.rnn(packed_input)
# 		# print(output_packed)
# 		out, lengths = rnn.pad_packed_sequence(output_packed)
# 		out = self.fc1(out)
# 		out = self.relu(out)
# 		out = self.do(out)
# 		# out = self.scoring(out)
# 		out = out.permute(1, 0, 2)
# 		# print(logits.shape)
# 		out = nn.AvgPool2d(kernel_size=(out.shape[1], 1))(out).squeeze()
# 		# print("---->", out.shape)
# 		# print(n_class_logits)
# 		# print(n_class_logits.shape)
# 		out = self.scoring(out)
# 		out = nn.functional.log_softmax(out, dim=1)
# 		# del input_features
# 		# for f in x:
# 		# 	del f
# 		# del packed_input
# 		# del output_packed
# 		# exit()
# 		# print(neu_sig)
# 		return out


# class CatEmotion(nn.Module):
# 	def __init__(self, class_n, hidden_size, nlayers ): # 47, 128, 1
# 		super(CatEmotion,self).__init__()
# 		self.embed = nn.Sequential(
# 			nn.Conv2d(1, 16, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False),
# 			nn.ReLU(),
# 			nn.Dropout(0.3),
# 			nn.Conv2d(16, 32, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False),
# 			nn.ReLU(),
# 			nn.Dropout(0.3),
# 			nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=(1, 1), bias=False),
# 			nn.ReLU(),
# 			nn.Dropout(0.3)
# 			)
# 		self.rnn = nn.LSTM(input_size = 40, hidden_size=hidden_size, num_layers=nlayers, bidirectional=True, dropout=0.1) # 1 layer, batch_size = False
# 		self.fc1 = nn.Linear(2*hidden_size,1024)
# 		self.do = nn.Dropout(p=0.2)
# 		self.relu = nn.ReLU()
# 		self.scoring = nn.Linear(1024,class_n)
# 		self.scoring2 = nn.Linear(1024, 2)

# 	def forward(self, seq_list):
# 		lens = [f.shape[0] for f in seq_list]
# 		features = [torch.from_numpy(f).cuda() for f in seq_list]
# 		input_features = rnn.pad_sequence(features)
# 		input_features = input_features.unsqueeze(1).permute(2,1,3,0) # (bs, 1, 40, max_len) bsx1xmax_lenx40
# 		# print("---->", input_features.size())
# 		# input_features = self.embed(input_features)
# 		# print("---->", input_features.size())
# 		# exit()

# 		n, c, h, w = input_features.size()
# 		input_features = input_features.view(n, c*h, w).permute(2, 0, 1)
# 		packed_input = rnn.pack_padded_sequence(input_features, lens) # packed version
# 		output_packed, hidden = self.rnn(packed_input)
# 		out, lengths = rnn.pad_packed_sequence(output_packed)
# 		out = self.fc1(out)
# 		out = self.relu(out)
# 		out = self.do(out)
# 		neu_sig = self.scoring2(out).permute(1, 0, 2)
# 		neu_sig = torch.sigmoid(nn.AvgPool2d(kernel_size=(neu_sig.shape[1], 1))(neu_sig).squeeze())

# 		n_class_logits = self.scoring(out)
# 		n_class_logits = n_class_logits.permute(1, 0, 2)
# 		# print(logits.shape)
# 		n_class_logits = nn.AvgPool2d(kernel_size=(n_class_logits.shape[1], 1))(n_class_logits).squeeze()
# 		# print("---->", out.shape)
# 		# print(n_class_logits)
# 		# print(n_class_logits.shape)
# 		n_class_logits = nn.functional.log_softmax(n_class_logits, dim=1)
# 		del input_features
# 		for f in features:
# 			del f
# 		del packed_input
# 		del output_packed
# 		# exit()
# 		# print(neu_sig)
# 		return neu_sig, n_class_logits, lengths

if __name__ == "__main__":
	model = multi_attention_model(768, 1, 2, [])
	# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	# sentence = ["my name is Hira", "I am you", "I"]
	# X_train = [tokenizer.tokenize('[CLS] ' + sent + ' [SEP]') for sent in sentence]
	# bert_model = BertModel.from_pretrained('bert-base-uncased')
	# bert_model = bert_model.cuda()
	# X_train_tokens = [tokenizer.convert_tokens_to_ids(sent) for sent in X_train]
	# results = torch.zeros((len(X_train_tokens), bert_model.config.hidden_size))#.long()
	# batch_size = 1
	# with torch.no_grad():
	#     for stidx in range(0, len(X_train_tokens), batch_size):
	#         X = X_train_tokens[stidx:stidx + batch_size]
	#         print(X)
	#         X = torch.LongTensor(X).cuda()
	#         _, pooled_output = bert_model(X)
	#         # print(pooled_output)
	#         results[stidx:stidx + batch_size,:] = pooled_output.cpu()
	#         print(X)
	#         print(pooled_output.cpu().shape)
			# print(results)
			# print(pooled_output.shape)
			# print(results.shape)
