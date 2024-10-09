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
from utils import Dataset, multi_attention_model_64, cuda, get_feature, emotions_used
# from pytorch_pretrained_bert import *
from bert_embedding import BertEmbedding
import argparse 
from joblib import load

class audio_model(nn.Module):
	def __init__(self):
		with open('target_vocab.txt', 'r') as fopen:
			target_vocab = fopen.readlines()
		## models loading 
		self.dtVmodel = load('models/dtV.model')
		self.dtAmodel = load('models/dtA.model')
		self.dtDmodel = load('models/dtD.model')

		model = multi_attention_model_64(768, 1, 4, target_vocab) # adding back the neutral class 
		device = torch.device("cuda:0" if cuda else "cpu")
		model = model.to(device).cuda() if cuda else model.to(device)
		# checkpoint = torch.load('/home/hyd/multi-hop-attention/code/models/0_14.model')
		MODEL_NAME = '/home/hyd/multi-hop-attention/code/models/0_1464dim.model'
		checkpoint = torch.load(MODEL_NAME)
		model.load_state_dict(checkpoint['model_state_dict'])
		emotions_used = ['ang', 'hap','sad', 'neu']

		bert_embedding = BertEmbedding()
		model.eval()
		emo, v, a, d = run_main(args.infile)
		print("Predicted emotion: %s, V: %.2f, A: %.2f, D: %.2f" %(emotions_used[emo[0]], v, a, d))
		self.model = model

	def get_transcript(self, audio_file):
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
		trans = "I" 
		return trans

	def forward(self, file):
		print("processing file: %s" %file )
		feature_input = get_feature(file).astype(np.float32)
		feature_input_expand = np.expand_dims(feature_input, 1)
		trans = get_transcript(file)
		embeds = bert_embedding([trans])
		embeds = list(map(lambda x: x[1], embeds))
		embeds = list(map(lambda x: torch.Tensor(x), embeds))

		with torch.no_grad():
			logits, logits_vad =  self.model(feature_input_expand, embeds)
			pred = torch.max(logits, dim=1)[1]

		dtVmodel_pred = np.mean(self.dtVmodel.predict(feature_input))
		dtAmodel_pred = np.mean(self.dtAmodel.predict(feature_input))
		dtDmodel_pred = np.mean(self.dtDmodel.predict(feature_input))
		return pred, np.mean([logits_vad[0,0].cpu().numpy(), dtVmodel_pred]), \
					 np.mean([logits_vad[0,1].cpu().numpy(), dtAmodel_pred]), \
					 np.mean([logits_vad[0,2].cpu().numpy(), dtDmodel_pred])

emotion_model = audio_model()
emotion_model.forward('ex.wav')