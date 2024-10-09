import numpy as np
import os, sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data.dataloader as dataloader
import torch.nn.utils.rnn as rnn
import time
import string
from bert_embedding import BertEmbedding
from python_speech_features import logfbank, fbank
from scipy.io import wavfile
import argparse
import bcolz, pickle
from joblib import dump, load
from main import IEMOCAP

cuda = True 
os.environ['DATA_PATH'] = '/home/hyd/multi-hop-attention/prepro_data'
BATCH_SIZE = 16
n_epochs = 15

glove_path = "/home/hyd/multi-hop-attention/data"
vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))
global new_word2idx
new_word2idx = {'PAD': 0}
cuda = True 

emotions_used = ['ang', 'hap','sad', 'neu']
bert_embedding = BertEmbedding()

ctc_path = "/home/hyd/DL-Spring19/Hw3/hw3p2/code/"
sys.path.insert(1, ctc_path)
from myCTC import *


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

class multi_attention_model(nn.Module):
	def __init__(self, hidden_size, nlayers, class_n, target_vocab):
		# target_vocab.append('PAD')
		global new_word2idx
		# print(target_vocab)
		super(multi_attention_model, self).__init__()
		# audioBRE = nn.ModuleList()
		self.audioBRE_lstm = nn.LSTM(input_size = 40, hidden_size=hidden_size, \
								num_layers=nlayers, bidirectional=True)
		self.audioBRE_fc = nn.Linear(2*hidden_size,class_n)
		# self.audioBRE_L = audioBRE

		weights_matrix, new_word2idx = init_embeding(target_vocab)
		# print(len(new_word2idx.keys()))
		embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True) 
		self.embedding = embedding
		self.textBRE_lstm = nn.LSTM(input_size = 768, hidden_size=hidden_size, \
								num_layers=nlayers, bidirectional=True)
		self.textBRE_fc =  nn.Linear(2*hidden_size,class_n)
		self.fc = nn.Linear(4*hidden_size,class_n)
		nn.init.xavier_uniform_(self.fc.weight)
		self.fc.bias.data.fill_(0.1)
		# self.audioBRE = nn.Sequential( *audioBRE)
		# self.textBRE = nn.Sequential( *textBRE)
		self.ctcmodel = CTCPhonemeSegmentor(47,512,3)
		self.ctcmodel = self.ctcmodel.cuda()
		checkpoint = torch.load('/home/hyd/for_shahan/14.model')
		self.ctcmodel.load_state_dict(checkpoint['model_state_dict'])

		self.fc_vad = nn.Linear(class_n, 3) #nn.Sequential(nn.LeakyReLU(), \ 
						
	def forward(self, x1, x2):
		'''
		x1 is the audio representation. 
		x2 is the indices of the words in the text
		'''
		print("---", x1.shape)
		exit()
		features_x1 = [torch.from_numpy(f).cuda() for f in x1] if cuda\
						else [torch.from_numpy(f) for f in x1]
		lens_x1 = torch.Tensor([f.shape[0] for f in x1]) # audio lens 
		lens_x2 = torch.Tensor([f.shape[0] for f in x2]) # trans lens 
		x2_sorted_lengths, order = torch.sort(lens_x2, 0, descending=True)
		_, backorder = torch.sort(order, 0)
		out = rnn.pad_sequence(features_x1).transpose(0, 1)
		audio_lstm, (audio_hidden, _) = self.audioBRE_lstm(out) 
		audio_hidden = audio_hidden[-2:, :, :]
		_, b, _ = audio_hidden.size()
		print(audio_hidden.size)
		audio_hidden = audio_hidden.transpose(0,1).reshape(b, -1)#.contiguous().view(b, -1)

		out = rnn.pad_sequence(x2)
		out = out.cuda() if cuda else out
		text_lstm, (text_hidden, _) = self.textBRE_lstm(out)

		_, b, _ = text_hidden.size()
		text_hidden = text_hidden.transpose(0,1).reshape(b, -1)
		out = torch.cat((text_hidden, audio_hidden), dim=1)

		out = self.fc(out)
		out_vad = self.fc_vad(out)
		out = nn.functional.log_softmax(out, dim=1)
		return out, out_vad


def get_feature(audio_file_name):
	'''
	calculates the mel log of the audio file 
	'''
	rate, data = wavfile.read(audio_file_name)
	output, _ = fbank(data,samplerate=rate, winlen=0.025625,
									  winstep=0.01, nfilt=40, nfft=512,
									  lowfreq=100, highfreq=3800, winfunc=np.hamming)
	output = np.log(output)
	return output.astype(np.float32)

def get_transcript(audio_file_name):
	'''
	if the data has transcripts, put here the exact way to extract them
	and return them as a string where each word is seperated with single
	space and the string is lower case. 
	Apply this function if possible
	sentence.replace("\n", "").lower().translate(str.maketrans('','',string.punctuation))
	If transcript is not available, then I will put here the Listen Attend Spell Model (LAS)
	which takes in audio and produces the transcript. However, its prone to spelling
	mistakes
	'''
	trans = "I am working right now please" 
	return trans


def collate_lines(seq_list):
	inputs, targets, trans, vad = zip(*seq_list)
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

	return inputs, targets, trans, vad, None

def get_VAD_from_audio(model, audio_file):
	model.eval()
	mel_log_feat = get_feature(audio_file)
	mel_log = np.expand_dims(mel_log_feat, 1)

	trans = get_transcript(audio_file)
	embeds = bert_embedding([trans])
	embeds = list(map(lambda x: x[1], embeds))
	embeds = list(map(lambda x: torch.Tensor(x), embeds))
	# embeds = torch.Tensor(embeds)#.unsqueeze(0)

	with torch.no_grad():
		logits, logits_vad = model(mel_log, embeds)
		pred_emotion = torch.max(logits, dim=1)[1]
		pred_v, pred_a, pred_d = logits_vad[:,0], logits_vad[:,1], logits_vad[:,2]

	print(mel_log_feat.shape)
	clf = load('dtV.model')
	clf_predict = clf.predict(mel_log_feat)
	print(np.mean(clf_predict))
	clf = load('dtA.model')
	clf_predict = clf.predict(mel_log_feat)
	print(np.mean(clf_predict))
	clf = load('dtD.model')
	clf_predict = clf.predict(mel_log_feat)
	print(np.mean(clf_predict))
	return pred_emotion, pred_v, pred_a, pred_d 

def get_VAD_from_feature(model, dtVmodel, dtAmodel, dtDmodel, mel_log_feat):
	model.eval()
	mel_log = np.expand_dims(mel_log_feat, 1)

	trans = get_transcript(audio_file)
	embeds = bert_embedding([trans])
	embeds = list(map(lambda x: x[1], embeds))
	embeds = list(map(lambda x: torch.Tensor(x), embeds))
	# embeds = torch.Tensor(embeds)#.unsqueeze(0)

	with torch.no_grad():
		logits, logits_vad = model(mel_log, embeds)
		pred_emotion = torch.max(logits, dim=1)[1]
		pred_v, pred_a, pred_d = logits_vad[:,0], logits_vad[:,1], logits_vad[:,2]

	
	dtVmodel_pred = np.mean(dtVmodel.predict(mel_log_feat))
	dtAmodel_pred = np.mean(dtAmodel.predict(mel_log_feat))
	dtDmodel_pred = np.mean(dtDmodel.predict(mel_log_feat))
	# print(pred_v.cpu().numpy()[0], dtVmodel_pred)
	# print(np.mean([pred_v.cpu().numpy()[0], dtVmodel_pred]))
	return pred_emotion, dtVmodel_pred, dtAmodel_pred, dtDmodel_pred
	return pred_emotion, np.mean([pred_v.cpu().numpy()[0], dtVmodel_pred]), \
						 np.mean([pred_a.cpu().numpy()[0], dtAmodel_pred]), \
						 np.mean([pred_d.cpu().numpy()[0], dtDmodel_pred])

def mae(a, b):
	return (np.sum(a - b) / len(a))

def mse(a, b):
	return (np.sum((a - b)**2) / len(a))

def rmse(a, b):
	return np.sqrt(np.sum((a - b)**2) / len(a))


def check_model_performance(model, dtVmodel, dtAmodel, dtDmodel):
	train_idx = []
	test_idx = []
	dev_idx = [9]
	loader = IEMOCAP(train_idx, test_idx, dev_idx)

	devX, devY, devTY, devVAD = loader.dev()
	# print(devX)
	pred_vl, pred_al, pred_dl = [], [], []
	for i in range(len(devX)):
		pred_emotion, pred_v, pred_a, pred_d = get_VAD_from_feature(model, dtVmodel, dtAmodel, dtDmodel, devX[i].astype(np.float32))
		# print("Predicted emotion: %s" %(emotions_used[pred_emotion]))
		# print("V: %.3f, A: %.3f, D: %.3f" %(pred_v, pred_a, pred_d))
		pred_vl.append(pred_v)
		pred_al.append(pred_a)
		pred_dl.append(pred_d)

	print(mse(pred_vl, devVAD[:, 0]))
	print(mse(pred_al, devVAD[:, 1]))
	print(mse(pred_dl, devVAD[:, 2]))
	return



if __name__ == '__main__':
	parser = argparse.ArgumentParser( prog='vad_wrapper.py', description='Takes in the input wav file and outputs \
																	(1) predicted emotion class \
																	(2) predicted values of V, A, D')

	parser.add_argument('-audio_file', required=True, help='The input audio file to process')
	args = parser.parse_args()

	audio_file = args.audio_file

	print("Processing audio file: %s ..." %(audio_file))
	
	with open('target_vocab.txt', 'r') as fopen:
		target_vocab = fopen.readlines()

	model = multi_attention_model(768, 1, 4, target_vocab) # 768 is the bert embedding dimension 
	device = torch.device("cuda:0" if cuda else "cpu")
	model = model.to(device).cuda() if cuda else model.to(device)
	# checkpoint = torch.load('/home/hyd/multi-hop-attention/code/models/0_14.model')
	# model.load_state_dict(checkpoint['model_state_dict'])

	# pred_emotion, pred_v, pred_a, pred_d = get_VAD_from_audio(model, devX)
	# print("Predicted emotion: %s" %(emotions_used[pred_emotion]))
	# print("V: %.3f, A: %.3f, D: %.3f" %(pred_v, pred_a, pred_d))
	dtVmodel = load('dtV.model')
	dtAmodel = load('dtA.model')
	dtDmodel = load('dtD.model')
	check_model_performance(model, dtVmodel, dtAmodel, dtDmodel)


	print("Done")