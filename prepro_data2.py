import os, sys
import numpy as np
from scipy.io import wavfile
from python_speech_features import logfbank, fbank
import pickle 
from transformers import Wav2Vec2Processor, HubertModel
import soundfile as sf
# from bert_embedding import BertEmbedding
# /bert_embedding = BertEmbedding()
from transformers import BertTokenizer, BertModel
import torch
import string, math


USE_VAD = False
WAV_DIR ="/home/hyd/workhorse2/multi-hop-attention/data/IEMOCAP_full_release/"
MELLOG_DIR  = "/home/hyd/workhorse2/multi-hop-attention/data/IEMOCAP_full_release_mellog/"
PHONE_DIR = "/home/hyd/workhorse2/multi-hop-attention/data/IEMOCAP_full_release/"
WORD_DIR = "/home/hyd/workhorse2/multi-hop-attention/data/IEMOCAP_full_release/"
SAVE_DIR = "../prepro_data_4classes-mellog40/"
MEL_OUT_DIR = "/home/hyd/workhorse2/multi-hop-attention/data/IEMOCAP_full_release_mellog_40/"
HUBERT_OUT_DIR='hubert_out/'
emotions_used = ['ang', 'hap', 'neu', 'sad']


bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")



def get_req_files(dirname, ext, special ):
	req_files = []
	for root, dirs, files in os.walk(dirname):
		if special in root:
			for name in files:
				if name.endswith(ext):
					req_files.append(os.path.join(root, name))
					# return req_files	
	return req_files


def get_bert_embeddings(trans):
	# import pdb 
	# pdb.set_trace()
	def remove_dup(trans):
		prev_wd = None
		new_trans = []
		for wd in trans:
			if prev_wd and wd == prev_wd:
				continue
			new_trans.append(wd)
			prev_wd=wd
		return (' ').join(new_trans)

	
	text = remove_dup(trans)
	inputs = bert_tokenizer(text, return_tensors="pt")
	outputs = bert_model(**inputs)

	last_hidden_states = outputs.last_hidden_state.squeeze().detach().numpy()

	return last_hidden_states

def get_bert_embeddings_extended(trans, xbert):
	# trans is the extended form 
	def remove_dup(trans):
		prev_wd = None
		new_trans = []
		for wd in trans:
			if prev_wd and wd == prev_wd:
				continue
			new_trans.append(wd)
			prev_wd=wd
		return new_trans
	
	

	trans2=(' ').join(trans).replace("\n", "").lower().translate(str.maketrans('','',string.punctuation))
	tokens=bert_tokenizer.basic_tokenizer.tokenize(trans2)
	inputs=bert_tokenizer.encode(tokens, return_tensors="pt")
	inputs = inputs[:,1:-1] # tokenizer adds start and end tokens 
	# print("1: ", inputs.shape)
	# import pdb 
	# pdb.set_trace()
	if inputs.shape[1] > 512:
		embeds_extented=[]
		for i in range(math.ceil(inputs.shape[1] / 512)):
			cuta, cutb =(i)*512,  (i+1)*512
			inputs1 = inputs[:,cuta:cutb]
			# inputs2 = inputs[:,cut:]
			outputs1 = bert_model(inputs1)
			embeds_extented1 = outputs1.last_hidden_state.squeeze().detach().numpy()
			# outputs2 = bert_model(inputs2)
			# embeds_extented2 = outputs2.last_hidden_state.squeeze().detach().numpy()
			# embeds_extented = np.concatenate((embeds_extented1, embeds_extented2), axis=0)
			if len(embeds_extented1.shape)==1:
				embeds_extented1 = embeds_extented1.reshape(1,-1)
			embeds_extented.append(embeds_extented1)
			# print(embeds_extented1.shape)
		embeds_extented=np.concatenate(embeds_extented)
	
	else:
		outputs = bert_model(inputs)
		embeds_extented = outputs.last_hidden_state.squeeze().detach().numpy()
	assert(len(trans) == len(embeds_extented))

	trans = remove_dup(trans)
	trans2=(' ').join(trans).replace("\n", "").lower().translate(str.maketrans('','',string.punctuation))
	tokens=bert_tokenizer.basic_tokenizer.tokenize(trans2)
	inputs=bert_tokenizer.encode(tokens, return_tensors="pt")
	inputs = inputs[:,1:-1] # tokenizer adds start and end tokens 
	# print("2: ", inputs.shape)
	outputs = bert_model(inputs)
	embeds = outputs.last_hidden_state.squeeze().detach().numpy()



	# def count_dup(trans):
	# 	counts = []
	# 	c = 1
	# 	prev_wd = trans[0]
	# 	for wd in trans[1:]:
	# 		if wd == prev_wd:
	# 			c += 1
	# 			prev_wd=wd
	# 			continue 
	# 		counts.append(c)
	# 		prev_wd=wd
	# 		c=1
	# 	counts.append(c)
	# 	return counts 

	# pdb.set_trace()
	
	# counts = count_dup(trans)
	# assert(len(counts) == len(xbert))
	# xbertnew = []
	# for i, c in enumerate(counts):
	# 	vec = xbert[i]
	# 	for j in range(c):
	# 		xbertnew.append(vec)
	# xbertnew = np.array(xbertnew)
	return embeds, embeds_extented

def get_phone_word(file):
	
	with open(file, 'r') as fopen:
		
		alllines = fopen.readlines()[1:-1] #skipping first and last line because header and footer information
		phone_extended = []
		for line in alllines:
			ssplit = line.split(' ')
			ssplit = list(filter(lambda l: (l !="" and l!="\t"), ssplit))
			s, e, phone = int(ssplit[0]), int(ssplit[1]), ssplit[3].rsplit()[0]
			ph = [phone]*(e-s+1)
			phone_extended.extend(ph)
	return phone_extended


def get_labels(phone_files_dict, word_file_dict, mellog_dict, hubert_dict, bertembeds_dict):
	d = {}
	with open("/home/hyd/workhorse2/emo_classifier/data/label/label_mapping_iemocap.pkl", 'rb') as f:
		label_dict = pickle.load(f)
	keys = list(label_dict.keys())
	lv_distance = 0
	count = 0
	all_emotions = []
	for k in keys:
		# print(k)
		emo = label_dict[k]['emotion']
		if emo in emotions_used:
			try:
				mlg = mellog_dict[k]
			except Exception as e:
				print("mlg not found")
			try:
				lbl = label_dict[k]
			except Exception as e:
				print("lbl not found")
			try:
				wdseg=word_file_dict[k]
			except Exception as e:
				print("wdseg not found")
			try:
				phseg=phone_files_dict[k]
			except Exception as e:
				print("phseg not found")
			try:
				hubert=hubert_dict[k]
			except Exception as e:
				print("hubert not found")
			try:
				bertemb=bertembeds_dict[k]
			except Exception as e:
				print("bertemb not found")
			
			d[k] = [mlg, emo, lbl, wdseg, phseg, hubert, bertemb]
	return d

def utt2spk(filename):
	'''Ses01F_impro01_F000.wav'''
	basename = os.path.basename(filename)
	ssplit = basename.split("_") 
	gender = ssplit[-1][0] #F
	return ssplit[0][:-1] + "_" + gender

def get_per_spk(data):
	'''
	data is dictionary with 
	filename as key and value as the tuple of mellog and labels 
	e.g
	{
			Ses04F_impro01_F021 : 
			[mellog, ang, [v, a, d], transcription, [phoneme_seg]], 
			...
			}
	'''
	d = {}
	keys = list(data.keys())
	for k in keys:
		v = data[k]
		spk = utt2spk(k)
		if spk in d.keys():
			d[spk].append((v[0],v[1], v[2], v[3], v[4], v[5], v[6]))
		else:
			d[spk] = [(v[0],v[1], v[2], v[3], v[4], v[5], v[6])]
	return d

def get_feature(file_name):
	rate, data = wavfile.read(file_name)
	output, _ = fbank(data,samplerate=rate, winlen=0.025625,
									  winstep=0.01, nfilt=40, nfft=512,
									  lowfreq=100, highfreq=3800, winfunc=np.hamming)
	# print("mellog output shape: ", output.shape)
	output = np.log(output)
	outfile = MEL_OUT_DIR + '/' + os.path.basename(file_name).replace('.wav', '.mellog') 
	# print(file_name, outfile)
	# print(output.shape)
	out_dir = os.path.dirname(outfile)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	# np.save(outfile, output)
	# exit()
	return output

def get_hubert_features(file_name):
	audio = sf.read(file_name)[0]
	input_values = hubert_processor(audio, sampling_rate=16000, return_tensors="pt").input_values  # Batch size 1
	hidden_states = hubert_model(input_values).last_hidden_state
	hidden_states = hidden_states.squeeze().detach().numpy()
	key=(".").join(file_namee.split('/')[-1].split('.')[:-1])
	outfile = HUBERT_OUT_DIR + '/' + key + ".npy"
	out_dir = os.path.dirname(outfile)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	np.save(outfile, hidden_states)
	return True 

def save_data(data_per_spk):
	keys = data_per_spk.keys()
	c = 0
	for k in keys:
		X, Y_emo, Y_vad, Y_trans, Y_trans_ctm, Y_phone, X_hubert, X_bert = [], [], [], [], [], [], [], []
		for l in data_per_spk[k]:
			X.append(l[0])
			Y_emo.append(l[1])
			Y_vad.append(l[2])
			Y_trans.append(l[3])
			Y_phone.append(l[4])
			X_hubert.append(l[5])
			X_bert.append(l[6])

		X = np.array(X)
		Y_emo = np.array(Y_emo)
		Y_vad = np.array(Y_vad)
		Y_trans = np.array(Y_trans)
		Y_phone = np.array(Y_phone)
		X_hubert = np.array(X_hubert)
		X_bert = np.array(X_bert)
		# print(X.shape, X[0].shape)
		# print(len(Y_emo), Y_emo[0])

		if not os.path.exists(SAVE_DIR):
			os.makedirs(SAVE_DIR)

		np.save(SAVE_DIR+"/X_"+str(c) , X)
		np.save(SAVE_DIR+"/Y_emo"+str(c) , Y_emo)
		np.save(SAVE_DIR+"/Y_vad"+str(c) , Y_vad)
		np.save(SAVE_DIR+"/Y_trans"+str(c) , Y_trans)
		np.save(SAVE_DIR+"/Y_phone"+str(c) , Y_phone)
		np.save(SAVE_DIR+"/X_hubert"+str(c) , X_hubert)
		np.save(SAVE_DIR+"/X_bert"+str(c) , X_bert)
		c += 1

def helper():
	'''
	Wrote this to get the order of the files saved. Being an idiot I didnt save 
	the name of the keys 
	'''
	d = {}
	
	with open("/home/hyd/workhorse2/emo_classifier/data/label/label_mapping_iemocap.pkl", 'rb') as f:
		label_dict = pickle.load(f)
	keys = list(label_dict.keys())
	for k in keys:
		emo = label_dict[k]['emotion']
		if emo in emotions_used:
			d[k]=[]

	keys = list(d.keys())
	d_spk = {}
	for k in keys:
		spk = utt2spk(k)
		if spk in d_spk: d_spk[spk].append(k)
		else: d_spk[spk]=[k]

	keys = d_spk.keys()
	c = 0
	for k in keys: # keys here are speakers
		towrite=[]
		for l in d_spk[k]:
			towrite.append(k +" " +l + "\n")
		fopen = open('order_'+str(c)+'.txt', 'w')
		for line in towrite:
			fopen.write(line)
		fopen.close()
		c+=1
	
	return 

def helper2():
	word_files = get_req_files(PHONE_DIR, ".wdseg", "sentences/ForcedAlignment")
	word_file_dict =  {os.path.splitext(os.path.basename(file))[0]:get_phone_word(file) for file in word_files}
	
	path = '/home/hyd/workhorse2/multi-hop-attention/prepro_data_4classes-mellog40/'
	
	
	
	for spk_id in range(10):
		# X_bert_spk = {}
		X_bert, X_bert_extended = [], []
		order = open('order_'+str(spk_id)+'.txt', 'r').readlines()
		temp_set_X_bert = np.load(os.path.join(path, "X_bert"+str(spk_id)+".npy"), allow_pickle=True)
		assert(len(order) == len(temp_set_X_bert))

		for o, bb in enumerate(temp_set_X_bert):

			order_split = order[o].split(' ')
			_, key = order_split[0], order_split[1].replace('\n', '')
			val = word_file_dict[key]
			bert, bert_extended = get_bert_embeddings_extended(val, bb)
			X_bert.append(bert)
			X_bert_extended.append(bert_extended)

		X_bert = np.array(X_bert)
		X_bert_extended = np.array(X_bert_extended)
		np.save(SAVE_DIR+"/X_bert_2"+str(spk_id) , X_bert)
		np.save(SAVE_DIR+"/X_bert_extended"+str(spk_id) , X_bert_extended)
	
	return 


if __name__ == "__main__":
	# fe = get_feature('/home/hyd/workhorse2/multi-hop-attention/data/IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav')
	# pg = get_phone_word('/home/hyd/workhorse2/multi-hop-attention/data/IEMOCAP_full_release/Session1/sentences/ForcedAlignment/Ses01F_impro01/Ses01F_impro01_F000.phseg')
	# wd = get_phone_word('/home/hyd/workhorse2/multi-hop-attention/data/IEMOCAP_full_release/Session1/sentences/ForcedAlignment/Ses01F_impro01/Ses01F_impro01_F000.wdseg')
	
	
	
	# print(fe)
	# print(pg)
	# print(wd)
	# print(fe.shape, len(pg), len(wd))
	# exit()
	# import pickle 
	# helper()

	helper2()
	exit()

	wav_files = get_req_files(WAV_DIR, ".wav", "wav/")
	mellog_files = get_req_files(MELLOG_DIR, ".npy", "")
	phone_files = get_req_files(PHONE_DIR, ".phseg", "sentences/ForcedAlignment")
	word_files = get_req_files(PHONE_DIR, ".wdseg", "sentences/ForcedAlignment")
	hubert_files = get_req_files(HUBERT_OUT_DIR, ".npy", "")

	print(len(wav_files), len(mellog_files), len(phone_files), len(word_files))

	

	# hubert_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
	# hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
	# hubert_dict = dict( ((".").join(file.split('/')[-1].split('.')[:-1]), get_hubert_features(file) ) for file in wav_files)

	phone_files_dict = {os.path.splitext(os.path.basename(file))[0]:get_phone_word(file) for file in phone_files}
	word_file_dict =  {os.path.splitext(os.path.basename(file))[0]:get_phone_word(file) for file in word_files}

	bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	bert_model = BertModel.from_pretrained("bert-base-uncased")
	bertembeds_dict = {}
	for key, val in word_file_dict.items():
		bertembeds_dict[key] = get_bert_embeddings(val)

		
	mellog_dict = dict( ((".").join(file.split('/')[-1].split('.')[:-2]), np.load(file).astype(np.float32)) for file in mellog_files)
	# print(mellog_dict)
	hubert_dict = dict( ((".").join(file.split('/')[-1].split('.')[:-1]), np.load(file).astype(np.float32) ) for file in hubert_files)
	

	 # = {os.path.splitext(os.path.basename(file))[0]:get_bert_embeddings(get_phone_word(file)) for file in word_files}
	
	data = get_labels(phone_files_dict, word_file_dict, mellog_dict, hubert_dict, bertembeds_dict)
	
	# pdb.set_trace()
	# exit()

	print(len(list(data.keys())))
	data_per_spk = get_per_spk(data)
	print(len(list(data_per_spk)))
	# print(data_per_spk)
	print("saving data")
	save_data(data_per_spk)
