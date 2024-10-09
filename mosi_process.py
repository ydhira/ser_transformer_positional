import os, sys
import numpy as np
from scipy.io import wavfile
from python_speech_features import logfbank, fbank
import pickle 
# from transformers import Wav2Vec2Processor, HubertModel
import soundfile as sf
# from bert_embedding import BertEmbedding
# /bert_embedding = BertEmbedding()
from transformers import BertTokenizer, BertModel
import torch
import string, math
import h5py

WAV_DIR = '/home/hyd/workhorse2/multi-hop-attention/code-clean/transformer/Raw/Audio/WAV_16000/Segmented/'
MEL_OUT_DIR = '/home/hyd/workhorse2/multi-hop-attention/code-clean/transformer/Raw/Mel/WAV_16000/Segmented/'
HUBERT_OUT_DIR = '/home/hyd/workhorse2/multi-hop-attention/code-clean/transformer/Raw/hubert_out/WAV_16000/Segmented/'
BERT_OUT_DIR = '/home/hyd/workhorse2/multi-hop-attention/code-clean/transformer/Raw/bert_out/WAV_16000/Segmented/'
LABEL_FILE = '/home/hyd/workhorse2/multi-hop-attention/code-clean/transformer/Raw/CMU_MOSI_Opinion_Labels.csd'  
PHONE_ALIGN_FILE = '/home/hyd/workhorse2/multi-hop-attention/code-clean/transformer/Raw/CMU_MOSI_TimestampedPhones.csd'
WORD_ALIGN_FILE = '/home/hyd/workhorse2/multi-hop-attention/code-clean/transformer/Raw/CMU_MOSI_TimestampedWords.csd'
SAVE_DIR = '/home/hyd/workhorse2/multi-hop-attention/code-clean/transformer/Raw/prepro_data/'

standard_train_fold=['2iD-tVS8NPw', '8d-gEyoeBzc', 'Qr1Ca94K55A', 'Ci-AH39fi3Y', '8qrpnFRGt2A', 'Bfr499ggo-0', 'QN9ZIUWUXsY', '9T9Hf74oK10', '7JsX8y1ysxY', '1iG0909rllw', 'Oz06ZWiO20M', 'BioHAh1qJAQ', '9c67fiY0wGQ', 'Iu2PFX3z_1s', 'Nzq88NnDkEk', 'Clx4VXItLTE', '9J25DZhivz8', 'Af8D0E4ZXaw', 'TvyZBvOMOTc', 'W8NXH0Djyww', '8OtFthrtaJM', '0h-zjBukYpk', 'Vj1wYRQjB-o', 'GWuJjcEuzt8', 'BI97DNYfe5I', 'PZ-lDQFboO8', '1DmNV9C1hbY', 'OQvJTdtJ2H4', 'I5y0__X72p0', '9qR7uwkblbs', 'G6GlGvlkxAQ', '6_0THN4chvY', 'Njd1F0vZSm4', 'BvYR0L6f2Ig', '03bSnISJMiM', 'Dg_0XKD0Mf4', '5W7Z1C_fDaE', 'VbQk4H8hgr0', 'G-xst2euQUc', 'MLal-t_vJPM', 'BXuRRbG0Ugk', 'LSi-o-IrDMs', 'Jkswaaud0hk', '2WGyTLYerpo', '6Egk_28TtTM', 'Sqr0AcuoNnk', 'POKffnXeBds', '73jzhE8R1TQ', 'OtBXNcAL_lE', 'HEsqda8_d0Q', 'VCslbP0mgZI', 'IumbAb8q2dM']

standard_valid_fold=['WKA5OygbEKI', 'c5xsKMxpXnc', 'atnd_PF-Lbs', 'bvLlb-M3UXU', 'bOL9jKpeJRs', '_dI--eQ6qVU', 'ZAIRrfG22O0', 'X3j2zQgwYgE', 'aiEXnCPZubE', 'ZUXBRvtny7o']

standard_test_fold=['tmZoasNr4rU', 'zhpQhgha_KU', 'lXPQBPVc5Cw', 'iiK8YX8oH1E', 'tStelxIAHjw', 'nzpVDcQ0ywM', 'etzxEpPuc6I', 'cW1FSBF59ik', 'd6hH302o4v8', 'k5Y_838nuGo', 'pLTX3ipuDJI', 'jUzDDGyPkXU', 'f_pcplsH_V0', 'yvsjCA6Y5Fc', 'nbWiPyCm4g0', 'rnaNMUZpvvg', 'wMbj6ajWbic', 'cM3Yna7AavY', 'yDtzw_Y-7RU', 'vyB00TXsimI', 'dq3Nf_lMPnE', 'phBUpBr1hSo', 'd3_k5Xpfmik', 'v0zCBqDeKcE', 'tIrG4oNLFzE', 'fvVhgmXxadc', 'ob23OKe5a9Q', 'cXypl4FnoZo', 'vvZ4IcEtiZc', 'f9O3YtZ2VfI', 'c7UH_rxdZv4']

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

def readdata():
	opinion_labels=h5py.File(LABEL_FILE)
	phonemes=h5py.File(PHONE_ALIGN_FILE)
	words=h5py.File(WORD_ALIGN_FILE)

	recordings = np.array(opinion_labels['Opinion Segment Labels/data'])
	utt_start_end_times = [np.array(opinion_labels['Opinion Segment Labels/data/'+r+'/intervals/']) for r in recordings] # start end times for each utterance within this recording
	emo_labels = [np.array(opinion_labels['Opinion Segment Labels/data/'+r+'/features/']) for r in recordings]
	
	word_start_end_times = [np.array(words['words/data/'+r+'/intervals/']) for r in recordings] 
	word_labels = [np.array(words['words/data/'+r+'/features/']) for r in recordings]

	phoneme_start_end_times = [np.array(phonemes['phoneme/data/'+r+'/intervals/']) for r in recordings] 
	phoneme_labels = [np.array(phonemes['phoneme/data/'+r+'/features/']) for r in recordings]

	########### PROCESSING FOR UTTERANCES #############

	emo_labels_dict = {}
	for i in range(len(recordings)):
		recording = recordings[i]
		for j in range(utt_start_end_times[i].shape[0]):
			utt = recording+'_'+str(j+1)
			emo_labels_dict[utt] = emo_labels[i][j][0]

	#print(emo_labels_dict)
	words_label_dict={}
	for i in range(len(recordings)):
		recording = recordings[i]

		times = utt_start_end_times[i]
		for j in range(times.shape[0]):
			utttime=times[j]
			uttstart, uttend = utttime[0], utttime[1]
			wordstart, wordend = word_start_end_times[i][:,0], word_start_end_times[i][:,1]
			idx=np.where((wordstart>=uttstart)&(wordend<=uttend))
			wordsutt=word_labels[i][idx]
			wordsduration=word_start_end_times[i][idx]
			utt = recording+'_'+str(j+1)
			wordsutt1=[]
			# print(wordsutt, wordsduration)
			for k, word1 in enumerate(wordsutt):
				duration=int((wordsduration[k][1]-wordsduration[k][0])*100)
				key=word1[0].decode("utf-8")
				key=key.replace('sp', 'sil')
				wordsutt1.extend([key]*duration)
			# if recording == "2WGyTLYerpo":
				# print(utt, wordsutt1, len(wordsutt1))
			# if len(wordsutt1) == 0:
				# print('Empty utt: ', utt)
			words_label_dict[utt] = wordsutt1
		# exit()

	phonemes_label_dict={}
	for i in range(len(recordings)):
		recording = recordings[i]
		times = utt_start_end_times[i]

		for j in range(times.shape[0]):
			utttime=times[j]
			uttstart, uttend = utttime[0], utttime[1]
			phonemestart, phonemeend = phoneme_start_end_times[i][:,0], phoneme_start_end_times[i][:,1]

			idx=np.where((phonemestart>=uttstart)&(phonemeend<=uttend))
			phonemeutt=phoneme_labels[i][idx]
			phonemeduration=phoneme_start_end_times[i][idx]
			utt = recording+'_'+str(j+1)
			
			phonemeutt1=[]
			for k, phone in enumerate(phonemeutt):
				duration=int((phonemeduration[k][1]-phonemeduration[k][0])*100)
				key=phone[0].decode("utf-8").upper()
				if len(key)>2:
					key=key[:-1]
				key=key.replace('SP', 'SIL')
				phonemeutt1.extend([key]*duration)
			# print(utt, wordsutt1, len(wordsutt1))
			# exit()
			# if len(phonemeutt1) == 0:
			# 	print('Empty phonemeutt1: ', utt)
			phonemes_label_dict[utt] = phonemeutt1	

	return emo_labels_dict, phonemes_label_dict, words_label_dict


def get_req_files(dirname, ext, special ):
	req_files = []
	for root, dirs, files in os.walk(dirname):
		if special in root:
			for name in files:
				if name.endswith(ext):
					req_files.append(os.path.join(root, name))
					# return req_files	
	return req_files

def get_bert_embeddings_extended(key, trans):
	print(key)
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
	# print(embeds.shape, embeds_extented.shape)
	# BERT_OUT_DIR
	outfile1 = BERT_OUT_DIR + '/' + key + "bert.npy"
	outfile2 = BERT_OUT_DIR + '/' + key + "bert_extend.npy"
	out_dir = os.path.dirname(outfile1)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	np.save(outfile1, embeds)
	np.save(outfile2, embeds_extented)
	return embeds, embeds_extented

def get_hubert_features(file_name):
	# print(file_name)
	audio = sf.read(file_name)[0]
	if len(audio.shape)==2: # some files are 2 channels, some are one 
		audio = np.mean(audio, axis=1)
	input_values = hubert_processor(audio, sampling_rate=16000, return_tensors="pt").input_values  # Batch size 1
	hidden_states = hubert_model(input_values).last_hidden_state
	hidden_states = hidden_states.squeeze().detach().numpy()
	key=(".").join(file_name.split('/')[-1].split('.')[:-1])
	# print(key)
	outfile = HUBERT_OUT_DIR + '/' + key + ".npy"
	out_dir = os.path.dirname(outfile)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	np.save(outfile, hidden_states)
	# exit()
	return True 

def save_data(hubert_dict, bert_dict, bert_extend_dict, emo_labels_dict, phonemes_label_dict):
	Y_emo_train, Y_phone_train, X_hubert_train, X_bert_train, X_bert_extended_train  = [], [], [], [], []
	Y_emo_test, Y_phone_test, X_hubert_test, X_bert_test, X_bert_extended_test = [], [], [], [], []

	utterances = hubert_dict.keys()
	print(utterances)
	for utt in utterances:
		key = ('_').join(utt.split('_')[:-1])
		
		if key in standard_train_fold:
			Y_emo_train.append(emo_labels_dict[utt])
			Y_phone_train.append(phonemes_label_dict[utt])
			X_hubert_train.append(hubert_dict[utt])
			X_bert_train.append(bert_dict[utt])
			X_bert_extended_train.append(bert_extend_dict[utt])

		if key in standard_test_fold + standard_valid_fold:
			Y_emo_test.append(emo_labels_dict[utt])
			Y_phone_test.append(phonemes_label_dict[utt])
			X_hubert_test.append(hubert_dict[utt])
			X_bert_test.append(bert_dict[utt])
			X_bert_extended_test.append(bert_extend_dict[utt])

	np.save(SAVE_DIR+"/Y_emo_train" , Y_emo_train)
	np.save(SAVE_DIR+"/Y_phone_train" , Y_phone_train)
	np.save(SAVE_DIR+"/X_hubert_train" , X_hubert_train)
	np.save(SAVE_DIR+"/X_bert_2_train", X_bert_train)
	np.save(SAVE_DIR+"/X_bert_extended_train", X_bert_extended_train)

	np.save(SAVE_DIR+"/Y_emo_test" , Y_emo_test)
	np.save(SAVE_DIR+"/Y_phone_test" , Y_phone_test)
	np.save(SAVE_DIR+"/X_hubert_test" , X_hubert_test)
	np.save(SAVE_DIR+"/X_bert_2_test" , X_bert_test)
	np.save(SAVE_DIR+"/X_bert_extended_test", X_bert_extended_test)
	return 

if __name__ == "__main__": 
	wav_files = get_req_files(WAV_DIR, ".wav", "")
	hubert_files = get_req_files(HUBERT_OUT_DIR, ".npy", "")
	bert_files = get_req_files(BERT_OUT_DIR, "bert.npy", "")
	bertextend_files = get_req_files(BERT_OUT_DIR, "extend.npy", "")
	# print(wav_files[:10])
	# print(len(wav_files))

	emo_labels_dict, phonemes_label_dict, words_label_dict = readdata()
	from more_itertools import unique_everseen
	print(words_label_dict['03bSnISJMiM_10'], len(words_label_dict['03bSnISJMiM_10']) )
	print(list(unique_everseen(words_label_dict['03bSnISJMiM_10'])))
	print(phonemes_label_dict['03bSnISJMiM_10'], len(phonemes_label_dict['03bSnISJMiM_10']))
	print(list(unique_everseen(phonemes_label_dict['03bSnISJMiM_10'])))
	exit()

	for badkey in ['2WGyTLYerpo_44', '5W7Z1C_fDaE_4', 'BI97DNYfe5I_21', 'BvYR0L6f2Ig_22', 'VCslbP0mgZI_1', 'W8NXH0Djyww_2', 'd6hH302o4v8_37']:
		emo_labels_dict.pop(badkey, None)
		phonemes_label_dict.pop(badkey, None)
		words_label_dict.pop(badkey, None)

	# bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	# bert_model = BertModel.from_pretrained("bert-base-uncased")

	# bert_dict = {key: get_bert_embeddings_extended(key, words_label_dict[key]) for key in words_label_dict.keys()}
	bert_dict = dict( ((".").join(file.split('/')[-1].split('.')[:-1]).replace('bert', ''), np.load(file).astype(np.float32) ) for file in bert_files)
	bert_extend_dict = dict( ((".").join(file.split('/')[-1].split('.')[:-1]).replace('bert_extend', ''), np.load(file).astype(np.float32) ) for file in bertextend_files)
	print(list(bert_dict.keys())[:5])
	print(list(bert_extend_dict.keys())[:5])
	# {  for key, trans in words_label_dict}
	# hubert_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
	# hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
	# hubert_dict = {os.path.splitext(os.path.basename(wav))[0]:get_hubert_features(wav) for wav in wav_files }
	hubert_dict = dict( ((".").join(file.split('/')[-1].split('.')[:-1]), np.load(file).astype(np.float32) ) for file in hubert_files)
	for badkey in ['2WGyTLYerpo_44', '5W7Z1C_fDaE_4', 'BI97DNYfe5I_21', 'BvYR0L6f2Ig_22', 'VCslbP0mgZI_1', 'W8NXH0Djyww_2', 'd6hH302o4v8_37']:	
		hubert_dict.pop(badkey, None)

	# save_data(hubert_dict, bert_dict, bert_extend_dict, emo_labels_dict, phonemes_label_dict)