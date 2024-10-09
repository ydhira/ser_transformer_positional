import os, sys
import numpy as np
from python_speech_features import logfbank, fbank
from scipy.io import wavfile
import pickle 
from multiprocessing import Process, Pool
# import stringdist

USE_VAD = False
# mel log , window shift = 100 ms, window size = 25 ms.
# IN_DIR = "/usr0/databases/IEMOCAP_full_release/"
IN_DIR="/home/hyd/workhorse2/multi-hop-attention/data/IEMOCAP_full_release/"
OUT_DIR  = "/home/hyd/workhorse2/multi-hop-attention/data/IEMOCAP_full_release_mellog/"
CTM_TRANSCRIPT_DIR = "/home/hyd/DL-Spring19/Hw3/hw3p2/code/IEMOCAP_CTM/"
# PHONE_SEG_DIR = "/home/hyd/aligner/SEG_PHONEMES" # phoneme segmentation 
# SAVE_DIR = "../prepro_data_8classes/"
# SAVE_DIR = "../prepro_data/" # without vad 4 classes 


## FOR VAD
if USE_VAD: 
	OUT_DIR  = "/usr0/databases/IEMOCAP_full_release_mellog/"
	PHONE_SEG_DIR = "/home/hyd/aligner/SEG_PHONEMES"
	SAVE_DIR = "../prepro_data_4classes/"


emotions_used = ['ang', 'hap', 'neu', 'sad', 'exc']
# emotions_used = ['ang', 'hap', 'fru', 'sur' 'fea', 'exc', 'dis', 'sad', 'neu']

def utt2spk(filename):
	'''Ses01F_impro01_F000.wav'''
	basename = os.path.basename(filename)
	ssplit = basename.split("_") 
	gender = ssplit[-1][0] #F
	return ssplit[0][:-1] + "_" + gender

def utt2session(filename):
	basename = os.path.basename(filename)
	ssplit = basename.split("_") 
	return ssplit[0][:-1]

def get_req_files(dirname, ext, special ):
	req_files = []
	for root, dirs, files in os.walk(dirname):
		# print(root)
		if special in root:
			# print(root)
			for name in files:
				if name.endswith(ext):
					req_files.append(os.path.join(root, name))

	return req_files

def get_feature(file_name):
	rate, data = wavfile.read(file_name)
	output, _ = fbank(data,samplerate=rate, winlen=0.025625,
									  winstep=0.01, nfilt=40, nfft=512,
									  lowfreq=100, highfreq=3800, winfunc=np.hamming)
	# print("mellog output shape: ", output.shape)
	output = np.log(output)
	outfile = file_name.replace(IN_DIR, OUT_DIR).replace('.wav', '.mellog') 
	print(file_name, outfile)
	print(output.shape)
	out_dir = os.path.dirname(outfile)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	np.save(outfile, output)
	return 

def mellog_pool():
	wav_files = get_req_files(IN_DIR, ".wav")	
	print(len(wav_files))
	print(wav_files[:5])
	# wav_files = [wav_files[0]]
	print(len(wav_files))
	pool = Pool()             
	pool.map(get_feature, wav_files)

def get_labels(wav_files, file_trans_dict, file_mellog_dict, file_trans_ctm_dict, file_phone_dict):
	d = {}
	with open("/home/hyd/emo_classifier/data/label/label_mapping_iemocap.pkl", 'rb') as f:
		label_dict = pickle.load(f)
	keys = list(label_dict.keys())
	lv_distance = 0
	count = 0
	all_emotions = []
	for k in keys:
		emo = label_dict[k]['emotion']
		if emo in emotions_used:
			all_emotions.append(emo)
			if emo == "exc":
				emo = "hap"
			# making this check bcoz after applying VAD, some didnt have mel log. Files beca,e zero size. 
			if k not in list(file_mellog_dict.keys()):
				continue
			## check here bcoz some files didnt get a ctm made
			if k.replace(".wav", "") not in list(file_trans_ctm_dict.keys()):
				trans_ctm = ""
			else:
				trans_ctm = file_trans_ctm_dict[k.replace(".wav", "")]
			## check here bcoz some files didnt get an alignement from HMM. 
			## the words are not present in vocabulary. ALOT of such files not made. 
			if k.replace(".wav", "") not in list(file_phone_dict.keys()):
				# phone_seg = None # pseg not created 
				continue
			else:
				phone_seg = file_phone_dict[k.replace(".wav", "")]
				# print(phone_seg)
			# print(file_mellog_dict[k].shape)
			# print(label_dict[k]['vad'])
			# print(file_trans_dict[k.replace(".wav", "")])
			d[k] = [file_mellog_dict[k], emo, label_dict[k]['vad'], file_trans_dict[k.replace(".wav", "")], trans_ctm, phone_seg]
			# print(file_trans_dict[k.replace(".wav", "")], trans_ctm )
			lv_distance += stringdist.levenshtein(file_trans_dict[k.replace(".wav", "")].lower(), trans_ctm.lower())
			count += 1

	print("lv distance between true and google api transcript: ", lv_distance / count)
	print(list(set(all_emotions)))
	return d

def get_transcripts(transcript_files):
	d = {}
	for f in transcript_files:
		with open(f, 'r') as fopen:
			line = fopen.readlines()
			# print(line)
			for l in line:
				lines = l.split(" ")
				filename = lines[0]
				trans = (" ").join(lines[2:])
				# print(filename)
				d[filename] = trans.replace("\n", "")
	return d

def get_mellog(mellog_files):
	d = {}
	for file in mellog_files:
		mellog = np.load(file).astype(np.float32)
		# print(mellog.shape)
		key = (".").join(file.split('/')[-1].split('.')[:-2])
		# print(key)
		d[key] = mellog
	return d

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
			d[spk].append((v[0], 
							[v[1], v[2], v[3], v[4], v[5]]))
		else:
			d[spk] = [(v[0], 
							[v[1], v[2], v[3], v[4], v[5]])]
	return d

def save_data(data_per_spk):
	keys = data_per_spk.keys()
	c = 0
	for k in keys:
		X, Y_emo, Y_vad, Y_trans, Y_trans_ctm, Y_phone = [], [], [], [], [], []
		for l in data_per_spk[k]:
			X.append(l[0])
			Y_emo.append(l[1][0])
			Y_vad.append(l[1][1])
			Y_trans.append(l[1][2])
			Y_trans_ctm.append(l[1][3])
			Y_phone.append(l[1][4])
		X = np.array(X)
		# print(Y_emo[0])
		Y_emo = np.array(Y_emo)
		Y_vad = np.array(Y_vad)
		Y_trans = np.array(Y_trans)
		Y_phone = np.array(Y_phone)
		print(X.shape, X[0].shape)
		print(len(Y_emo), Y_emo[0])
		np.save(SAVE_DIR+"/X_"+str(c) , X)
		np.save(SAVE_DIR+"/Y_emo"+str(c) , Y_emo)
		np.save(SAVE_DIR+"/Y_vad"+str(c) , Y_vad)
		np.save(SAVE_DIR+"/Y_trans"+str(c) , Y_trans)
		np.save(SAVE_DIR+"/Y_trans_ctm"+str(c) , Y_trans_ctm)
		np.save(SAVE_DIR+"/Y_phone"+str(c) , Y_phone)
		c += 1

def get_txt_ctm(lines):
	filename = lines[0].split(" ")[0]
	trans = ""
	for line in lines:
		trans += " " + line.split(" ")[4].replace("\n", "")
	return filename, trans

def get_transcript_frm_ctm(ctm_files):
	'''
	ctm files are made form google cloud api transcripts. 
	This gather them and returns dictionary of filename: google trans
	'''
	d = {}
	for f in ctm_files:
		with open(f, 'r') as fopen:
			lines = fopen.readlines()
			filename, trans = get_txt_ctm(lines)
			d[filename] = trans.lower().replace('.', " ")

	return d

def get_pseg(lines):
	l = []
	for line in lines:
		line_split = line.split(" ")
		p, s, e = line_split[0][:-1], int(line_split[1][:-1]), int(line_split[2])
		l.append((p, s, e))
	return l

def get_phones_frm_files(pseg_files):
	'''
	phoneme segmation files are made from HMM aligner code 
	I got from prof Rita. 
	return:
		filename: [(phone: start, end )](all are string) 
		start and end are frame numbers 
	'''
	d = {}
	for f in pseg_files:
		filename = f.split('/')[-1].split('.')[0]
		with open(f, 'r') as fopen:
			lines = fopen.readlines()
			pseg = get_pseg(lines)
			d[filename] = pseg
			# print(pseg)
	return d

if __name__ == '__main__':
	#######################################
	with open("/home/hyd/workhorse2/emo_classifier/data/label/label_mapping_iemocap.pkl", 'rb') as f:
		label_dict = pickle.load(f)
	keys = list(label_dict.keys())
	# print(keys)
	# d = {}
	# for k in keys:
	# 	spk = utt2spk(k)
	# 	gender = spk.split("_")[1]
	# 	if spk in d.keys():
	# 		d[spk]+=1
	# 	else:
	# 		print(spk)
	# 		d[spk]=1
	########################################
	# mellog_pool()
	# 
	if USE_VAD:
		wav_files = get_req_files(IN_DIR, ".wav", "wav_vad") # for vad data 
	else:
		wav_files = get_req_files(IN_DIR, ".wav", "wav/")
	transcript_files = get_req_files(IN_DIR, ".txt", "transcriptions")
	mellog_files = get_req_files(OUT_DIR, ".npy", "")
	ctm_transcript_files = get_req_files(CTM_TRANSCRIPT_DIR, ".ctm", "")
	pseg_files = get_req_files(PHONE_SEG_DIR, ".pseg", "")
	
	file_trans_dict = get_transcripts(transcript_files)
	file_mellog_dict = get_mellog(mellog_files)
	file_trans_ctm_dict = get_transcript_frm_ctm(ctm_transcript_files)
	file_phone_dict = get_phones_frm_files(pseg_files)

	print(len(transcript_files))
	print(len(ctm_transcript_files))
	print(len(pseg_files))
	print(len(list(file_trans_dict.keys())))
	print(len(list(file_trans_ctm_dict.keys())))
	print(len(wav_files))
	data = get_labels(wav_files, file_trans_dict, file_mellog_dict, file_trans_ctm_dict, file_phone_dict)
	
	# exit()

	print(len(list(data.keys())))
	data_per_spk = get_per_spk(data)
	print(len(list(data_per_spk)))
	# print(data_per_spk)
	print("saving data")
	for k in data_per_spk.keys():
		print(k)
	# save_data(data_per_spk)
	# print(len(set(list(map(lambda x: utt2session(x), wav_files)))))
	# print(len(set(list(map(lambda x: utt2spk(x), wav_files)))))

	
