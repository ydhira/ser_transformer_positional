
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data.dataloader as dataloader
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from new_utils import Dataset, cuda, self_attn_model, emotions_used, Net, IEMOCAP, CMU_MOSI, load_raw, collate_lines, get_metrics
import os, sys
from tqdm import tqdm
from scipy.io import savemat, loadmat
import numpy as np
import string 
from transformer_class import TransformerClassAudio, TransformerClassAudioText
import matplotlib.pyplot as plt
import pdb 

speaker1_models={'A': 'models/1_49transformer-bs6-a_inputspace-conformer_adam_hubert_spk1.model',
				'B': 'models/1_49transformer_bs6_a_phfeatspace_aligned.model',
				'C': 'models/1_49transformer_bs6_a_phfeatspace_aligned_2.model',
				'D': 'models/1_49transformer_bs6_a_phfeat_unaligned.model',
				'E': 'models/1_49transformer_bs6_a_phfeat_aligned_unaligned.model',
				'F': 'models/1_49transformer_bs6_a_phfeat_aligned_unaligned-2.model'}

speaker2_models={'A': 'models/2_49transformer-bs6-a_inputspace-conformer_adam_hubert_spk2.model',
				'B': 'models/2_49transformer-bs6-a_phfeatspace_aligned-conformer_adam_hubert_spk2.model',
				'C': 'models/2_49transformer_bs6_a_phfeatspace_aligned_2.model',
				'D': 'models/2_49transformer_bs6_a_phfeat_unaligned.model',
				'E': 'models/2_49transformer_bs6_a_phfeat_aligned_unaligned.model',
				'F': 'models/2_49transformer_bs6_a_phfeat_aligned_unaligned-2.model'}

speaker3_models={'A': 'models/3_49transformer-bs6-a_inputspace-conformer_adam_hubert_spk3.model',
				'B': 'models/3_49transformer_bs6_a_phfeatspace_aligned.model',
				'C': 'models/3_49transformer_bs6_a_phfeatspace_aligned_2.model',
				'D': 'models/3_49transformer_bs6_a_phfeat_unaligned.model',
				'E': 'models/3_49transformer_bs6_a_phfeat_aligned_unaligned.model',
				'F': 'models/3_49transformer_bs6_a_phfeat_aligned_unaligned-2.model'}

speaker4_models={'A': 'models/4_49transformer-bs6-a_inputspace-conformer_adam_hubert_spk4.model',
				'B': 'models/4_49transformer-bs6-a_phfeatspace_aligned-conformer_adam_hubert_spk4.model',
				'C': 'models/4_49transformer_bs6_a_phfeatspace_aligned_2.model',
				'D': 'models/4_49transformer_bs6_a_phfeat_unaligned.model',
				'E': 'models/4_49transformer_bs6_a_phfeat_aligned_unaligned.model',
				'F': 'models/4_49transformer_bs6_a_phfeat_aligned_unaligned-2.model'}

speaker9_models={'A': 'models/9_49transformer-bs6-a_inputspace-conformer_adam_hubert_spk9.model',
				'B': 'models/9_49transformer-bs6-a_phfeatspace_aligned-conformer_adam_hubert_spk9.model', 
				'C': 'models/9_49transformer-bs6-a_phfeatspace_aligned_2-conformer_adam_hubert_spk9.model', #
				'D': 'models/9_49transformer-bs6-a_phfeat_unaligned-conformer_adam_hubert_spk9.model',
				'E': 'models/9_49transformer-bs6-a_phfeat_aligned_unaligned-conformer_adam_hubert_spk9.model',
				'F': 'models/9_49transformer-bs6-a_phfeat_aligned_unaligned-2-conformer_adam_hubert_spk9.model'}

###################333 CHANGING TO CONFORMER AND CHANAGING THE DIM #################################
speaker1_models={'A': 'models/1_49transformer-bs6-a_inputspace-conformer_adam_hubert_iemocap.model',
				'B': 'models/1_49transformer-bs6-a_phfeatspace_aligned-conformer_adam_hubert_iemocap.model', 
				'C': 'models/1_49transformer-bs6-a_phfeatspace_aligned_2-conformer_adam_hubert_iemocap.model',
				'D': 'models/1_49transformer-bs6-a_phfeat_unaligned-conformer_adam_hubert_iemocap.model',
				'E': 'models/1_49transformer-bs6-a_phfeat_aligned_unaligned-conformer_adam_hubert_iemocap.model',
				'F': 'models/1_49transformer-bs6-a_phfeat_aligned_unaligned-2-conformer_adam_hubert_iemocap.model'}

speaker2_models={'A': 'models/2_49transformer-bs6-a_inputspace-conformer_adam_hubert_iemocap.model',
				'B': 'models/2_49transformer-bs6-a_phfeatspace_aligned-conformer_adam_hubert_iemocap.model',
				'C': 'models/2_49transformer-bs6-a_phfeatspace_aligned_2-conformer_adam_hubert_iemocap.model',
				'D': 'models/2_49transformer-bs6-a_phfeat_unaligned-conformer_adam_hubert_iemocap.model',
				'E': 'models/2_49transformer-bs6-a_phfeat_aligned_unaligned-conformer_adam_hubert_iemocap.model',
				'F': 'models/2_49transformer-bs6-a_phfeat_aligned_unaligned-2-conformer_adam_hubert_iemocap.model'}

speaker3_models={'A': 'models/3_49transformer-bs6-a_inputspace-conformer_adam_hubert_iemocap.model',
				'B': 'models/3_49transformer-bs6-a_phfeatspace_aligned-conformer_adam_hubert_iemocap.model',
				'C': 'models/3_49transformer-bs6-a_phfeatspace_aligned_2-conformer_adam_hubert_iemocap.model',
				'D': 'models/3_49transformer-bs6-a_phfeat_unaligned-conformer_adam_hubert_iemocap.model',
				'E': 'models/3_49transformer-bs6-a_phfeat_aligned_unaligned-conformer_adam_hubert_iemocap.model',
				'F': 'models/3_49transformer-bs6-a_phfeat_aligned_unaligned-2-conformer_adam_hubert_iemocap.model'}

speaker4_models={'A': 'models/4_49transformer-bs6-a_inputspace-conformer_adam_hubert_iemocap.model',
				'B': 'models/4_49transformer-bs6-a_phfeatspace_aligned-conformer_adam_hubert_iemocap.model',
				'C': 'models/4_49transformer-bs6-a_phfeatspace_aligned_2-conformer_adam_hubert_iemocap.model',
				'D': 'models/4_49transformer-bs6-a_phfeat_unaligned-conformer_adam_hubert_iemocap.model',
				'E': 'models/4_49transformer-bs6-a_phfeat_aligned_unaligned-conformer_adam_hubert_iemocap.model',
				'F': 'models/4_49transformer-bs6-a_phfeat_aligned_unaligned-2-conformer_adam_hubert_iemocap.model'}

speaker9_models={'A': 'models/9_49transformer-bs6-a_inputspace-conformer_adam_hubert_iemocap.model',
				'B': 'models/9_49transformer-bs6-a_phfeatspace_aligned-conformer_adam_hubert_iemocap.model', 
				'C': 'models/9_49transformer-bs6-a_phfeatspace_aligned_2-conformer_adam_hubert_iemocap.model', #
				'D': 'models/9_49transformer-bs6-a_phfeat_unaligned-conformer_adam_hubert_iemocap.model', 
				'E': 'models/9_49transformer-bs6-a_phfeat_aligned_unaligned-conformer_adam_hubert_iemocap.model',
				'F': 'models/9_49transformer-bs6-a_phfeat_aligned_unaligned-2-conformer_adam_hubert_iemocap.model'}

cmu_mosi_models={'A': 'models/-1_49transformer-bs6-a_inputspace-conformer_adam_hubert_cmu-mosi.model',
				# 'F': 'models/-1_49transformer-bs6-a_phfeat_aligned_unaligned-2-conformer_adam_hubert_cmu-mosi.model',
				'E': 'models/-1_49transformer-bs6-a_phfeat_aligned_unaligned-conformer_adam_hubert_cmu-mosi.model',
				'C': 'models/-1_49transformer-bs6-a_phfeatspace_aligned_2-conformer_adam_hubert_cmu-mosi.model',
				'B':  'models/-1_49transformer-bs6-a_phfeatspace_aligned-conformer_adam_hubert_cmu-mosi.model',
				'D': 'models/-1_49transformer-bs6-a_phfeat_unaligned-conformer_adam_hubert_cmu-mosi.model'
				}

speaker_models = {1: speaker1_models, 2: speaker2_models, 3: speaker3_models, 4: speaker4_models, 9: speaker9_models}
speaker_mosi_models = {1: cmu_mosi_models}
modalities = ['a_inputspace',\
			'a_phfeatspace_aligned', 
			'a_phfeatspace_aligned_2', 
			'a_phfeat_unaligned',\
			'a_phfeat_aligned_unaligned', \
			'a_phfeat_aligned_unaligned-2' ]

modalities = ['a_phfeatspace_aligned']
model_name_dict = {'A': 'a_inputspace', 'B': 'a_phfeatspace_aligned',\
					'C': 'a_phfeatspace_aligned_2', 'D': 'a_phfeat_unaligned','E': 'a_phfeat_aligned_unaligned', 'F': 'a_phfeat_aligned_unaligned-2'}

model_name_dict_rev = {value: key for key, value in model_name_dict.items()}
# for speaker_models in [speaker1_models, speaker2_models, speaker3_models, speaker4_models, speaker9_models]:
# 	for modal, fname in speaker1_models.items():
# 		isFile = os.path.isfile(fname) 
# 		print(fname, isFile)

def passmodel(model, mod, batch_size, loader):
	## desceibe the model 
	
	features = []
	labels = []
	all_logits = []
	c = 0
	for x1, target, x2, vad_target, X_bertpad, X_bert_len, phone_target, input_lens, \
			phone_target_sequence, X_bert_extendpad, X_bert_extend_len, phone_target_len , phone_target_sequence_len in tqdm(loader):
		
		# input_lens = torch.Tensor(input_lens).cuda()
		# X_bert_len = torch.Tensor(X_bert_len).cuda()
		# X_bert_extend_len = torch.Tensor(X_bert_extend_len).cuda()
		# phone_target_len = torch.Tensor(phone_target_len).cuda()
		# phone_target_sequence_len = torch.Tensor(phone_target_sequence_len).cuda()

		with torch.no_grad():
			if mod == 'a_inputspace':
				feature, logits = model(x1,input_lens)
			if mod == "a_phfeatspace_aligned":
				feature, logits, (attention_lst_audio, attention_lst_phone, attention_lst_trans)= model(src=x1, txt=phone_target, trans=X_bert_extendpad , wav_len=input_lens, bert_len=X_bert_extend_len)
				# logits = model(x1, phone_target, input_lens)
				pdb.set_trace()
				fig = plt.figure()
				plt.imshow(attention_lst_audio[0][0].cpu().numpy())
				fig.savefig('attention_audio.png')
				plt.close()
				plt.imshow(attention_lst_phone[0][0].cpu().numpy())
				fig.savefig('attention_phone.png')
				plt.close()
				plt.imshow(attention_lst_trans[0][0].cpu().numpy())
				fig.savefig('attention_trans.png')
				plt.close()
			if mod == "a_phfeatspace_aligned_2":
				feature, logits = model.module.forward2(src=x1, txt=phone_target, trans=X_bert_extendpad , wav_len=input_lens, bert_len=X_bert_extend_len)
			if mod == "a_phfeat_unaligned":
				feature, logits = model.module.forward_unaligned(x1, phone_target_sequence, X_bert_extendpad, input_lens, X_bert_len)
			if mod == "a_phfeat_aligned_unaligned":
				feature, logits = model.module.forward_aligned_unaligned(src=x1, txt_aligned=phone_target, txt_unaligned=phone_target_sequence,\
					trans_aligned=X_bert_extendpad , trans_unaligned = X_bertpad, \
					wav_len=input_lens, trans_aligned_len=X_bert_extend_len, trans_unaligned_len = X_bert_len, \
					phone_aligned_len= phone_target_len, phone_unaligned_len = phone_target_sequence_len)
			if mod == "a_phfeat_aligned_unaligned-2":
				feature, logits = model.module.forward_aligned_unaligned2(src=x1, txt_aligned=phone_target, txt_unaligned=phone_target_sequence,\
					trans_aligned=X_bert_extendpad , trans_unaligned = X_bertpad, \
					wav_len=input_lens, trans_aligned_len=X_bert_extend_len, trans_unaligned_len = X_bert_len, \
					phone_aligned_len= phone_target_len, phone_unaligned_len = phone_target_sequence_len)

			features.append(feature)
			labels.extend(target)
			all_logits.append(logits)

	features = torch.cat(features, dim=0).squeeze().cpu().numpy() # (B, Feat_dim)
	all_logits = torch.cat(all_logits, dim=0)#.squeeze().cpu().numpy() #B, n_class
	print("After passing data through model: ", features.shape, all_logits.shape )
	return features, labels, all_logits

def init_model(mod, model_name, classn):
	encoder_module ='conformer'

	if mod == "a_inputspace": # A
		model = TransformerClassAudio(
				#tgt_vocab, input_size, d_model,d_pos_enc, nhead, num_encoder_layer, num_decoder_layer, d_ffn, n_class
				720, 1024, 64, 64, 8, 1, 1, 64, classn, activation=torch.nn.GELU, encoder_module=encoder_module, normalize_before=True
			)
	if mod == "a_phfeatspace_aligned": # B
		model = TransformerClassAudioText(
				#tgt_vocab, input_size, d_model, d_pos_enc, nhead, num_encoder_layer, num_decoder_layer, d_ffn, n_class
				720, 1024, 64, 64, 8, 1, 1, 64, classn, modalities=mod, activation=torch.nn.GELU, encoder_module=encoder_module, normalize_before=True
			)
	if mod == "a_phfeatspace_aligned_2": # C
		model = TransformerClassAudioText(
				#tgt_vocab, input_size, d_model, d_pos_enc, nhead, num_encoder_layer, num_decoder_layer, d_ffn, n_class
				720, 1024, 64, 64*3, 8, 1, 1, 64, classn, modalities=mod, activation=torch.nn.GELU, encoder_module=encoder_module, normalize_before=True
			)
	if mod == "a_phfeat_unaligned": # D
		model = TransformerClassAudioText(
				#tgt_vocab, input_size, d_model, d_pos_enc, nhead, num_encoder_layer, num_decoder_layer, d_ffn, n_class
				720, 1024, 64, 64, 8, 1, 1, 64, classn, modalities=mod, activation=torch.nn.GELU, encoder_module=encoder_module, normalize_before=True
			)
	if mod == "a_phfeat_aligned_unaligned": # E
		model = TransformerClassAudioText(
				#tgt_vocab, input_size, d_model, d_pos_enc, nhead, num_encoder_layer, num_decoder_layer, d_ffn, n_class
				720, 1024, 64, 64, 8, 1, 1, 64, classn, modalities=mod, activation=torch.nn.GELU, encoder_module=encoder_module, normalize_before=True
			)
	if mod == "a_phfeat_aligned_unaligned-2": # F
		model = TransformerClassAudioText(
				#tgt_vocab, input_size, d_model, d_pos_enc, nhead, num_encoder_layer, num_decoder_layer, d_ffn, n_class
				720, 1024, 64, 64*3, 8, 1, 1, 64, classn, modalities=mod, activation=torch.nn.GELU, encoder_module=encoder_module, normalize_before=True
			)

	device = torch.device("cuda:0" if cuda else "cpu")
	model = model.to(device).cuda() if cuda else model.to(device)
	model = nn.DataParallel(model, device_ids=[0,1])

	checkpoint = torch.load(model_name)

	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()
	return model 

def main(test_speaker):
	batch_size=6
	idx = np.array([0,1,2,3,4,5,6,7,8,9])
	test_idx = [test_speaker]
	dev_idx = [test_speaker]
	train_idx = np.setdiff1d(idx, test_idx + dev_idx)
	dataset='IEMOCAP'

	if dataset == 'CMU_MOSI':
		loader = CMU_MOSI()
		devX, devY, devTY, devVAD, devPhone, devBert, devBertExtend = loader.test()
		classn=2

	# trainX, trainY, trainTY, trainVAD, trainPhone, trainBert, trainBertExtend = loader.train()
	if dataset == 'IEMOCAP':
		loader = IEMOCAP(train_idx, test_idx, dev_idx)
		devX, devY, devTY, devVAD, devPhone, devBert, devBertExtend, _ = loader.dev()
		classn=4
		devTY = list(map(lambda sentence: (' ').join(sentence).replace("\n", "").lower().translate(str.maketrans('','',string.punctuation)) , devTY))
	# trainTY = list(map(lambda sentence: (' ').join(sentence).replace("\n", "").lower().translate(str.maketrans('','',string.punctuation)) , trainTY))

	# train = Dataset(trainX, trainY, trainTY, trainVAD, trainPhone, trainBert, trainBertExtend)
	dev = Dataset(devX, devY, devTY, devVAD, devPhone, devBert, devBertExtend, dataset=dataset)

	dataloader_args = dict(shuffle=False, batch_size=batch_size, collate_fn = collate_lines, num_workers=0)
	# train_loader = dataloader.DataLoader(train, **dataloader_args)
	dev_loader = dataloader.DataLoader(dev, **dataloader_args)
	
	all_logits = []

	for mod in modalities:
		for epoch_num in [49]:#[45, 47, 48, 49]:
			model_name = speaker_models[test_speaker][model_name_dict_rev[mod]] # for iemocap
			# if model_name_dict_rev[mod] not in speaker_mosi_models[test_speaker]:
				# continue
			# model_name = speaker_mosi_models[test_speaker][model_name_dict_rev[mod]]  
			# print("Loading ... ", model_name)
			model_name = model_name.replace('49', str(epoch_num))
			print("Loading model: ...  ", model_name)
			model = init_model(mod, model_name, classn)
			features, labels, logits = passmodel(model, mod, batch_size, dev_loader)
			all_logits.append(logits)

	# import pdb 
	# pdb.set_trace()
	all_logits = torch.stack(all_logits) # n_models, B, n_class
	# print("Logits after passign through all models: ", all_logits.shape)
	# avg_logits,_ = torch.mode(all_logits, dim=0) #B, n_class
	avg_logits = torch.mean(all_logits, dim=0)
	# print("Logits after avg: ", avg_logits.shape)

	pred = torch.max(avg_logits, dim=1)[1]
	# print("Pred: ", pred.shape)
	all_predictions = pred.detach().cpu().numpy()
	metrics = get_metrics(all_predictions, labels, 'test')
	print("Test - speaker: ", test_speaker)
	print(metrics)
	return 

if __name__ == "__main__":
	for test_speaker in [1,2,3,4,9]:
		main(test_speaker)