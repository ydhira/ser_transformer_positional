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
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from new_utils import Dataset, cuda, self_attn_model, emotions_used, Net, IEMOCAP, CMU_MOSI, load_raw, collate_lines, collate_lines_context, get_metrics, DatasetContext
# from pytorch_pretrained_bert import *
# from transformers import TransformerASR
from transformer_class import TransformerClassAudio, TransformerClassAudioText

import logging 
import wandb




# os.environ['DATA_PATH'] = '/home/hyd/multi-hop-attention/prepro_data'
# os.environ['DATA_PATH'] = '/home/hyd/workhorse2/multi-hop-attention/prepro_data_4classes-mellog40/' # accidently write 4 clases data into 8 classes

# emotions_used = ['ang', 'hap','sad', 'neu'] #previosuly used emotion classes 
# emotions_used = ['ang', 'hap', 'fru', 'sur', 'fea', 'exc', 'dis', 'sad', 'neu']


speakers={0:'Ses05_M',1:'Ses01_F',2:'Ses05_F',3:'Ses01_M',4:'Ses03_M',5:'Ses02_M',6:'Ses02_F',7:'Ses04_M',8:'Ses03_F',9:'Ses04_F'}




def val_model_trans(model, loader):
	model.eval()
	num_classes = classn #len(emotions_used)
	cm = np.zeros(shape=(num_classes, num_classes))
	attention, phone_segmented, emo_target = [], [] , []
	all_predictions = []
	y_test = []
	with torch.no_grad():
		# model.to(device)
		total_loop = 0
		correct = 0
		all_feats = []

		for x1, target, x2, vad_target, X_bertpad, X_bert_len, phone_target,input_lens,\
				phone_target_sequence, X_bert_extendpad, X_bert_extend_len, phone_target_len , phone_target_sequence_len in tqdm(loader):
		# for items in tqdm(loader):
			if len(x1) < 2:
				continue
			# target = torch.LongTensor(target).cuda() if cuda else torch.LongTensor(target)
			# vad_target = torch.FloatTensor(vad_target).cuda() if cuda else torch.FloatTensor(vad_target)
			# If using this. check the error TypeError: must be real number, not dict
			# target = items[2][1]
			# phone_target = items[2][6]
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

			if model_type == "transformer":
				# input_lens = torch.Tensor(input_lens).cuda()
				# X_bert_len = torch.Tensor(X_bert_len).cuda()
				# X_bert_extend_len = torch.Tensor(X_bert_extend_len).cuda()
				# phone_target_len = torch.Tensor(phone_target_len).cuda()
				# phone_target_sequence_len = torch.Tensor(phone_target_sequence_len).cuda()

				if modalities == 'a_inputspace':
					_,logits = model(x1,input_lens)
				if modalities=='a_ph_inputspace':
					x1=torch.cat((x1, phone_target),2)
					logits = model(x1,input_lens)

				if modalities == "a_phfeatspace_aligned":
					#src, txt, trans, wav_len=None, bert_le
					# print(x1.shape, X_bertpad)
					_, logits = model(src=x1, txt=phone_target, trans=X_bert_extendpad , wav_len=input_lens, bert_len=X_bert_extend_len)
					##model(items)
					# 
					# all_f/eats.extend(feats)

				if modalities == "a_phfeatspace_aligned_2":
					_,logits = model.module.forward2(src=x1, txt=phone_target, trans=X_bert_extendpad , wav_len=input_lens, bert_len=X_bert_extend_len)
				
				if modalities == "a_phfeat_unaligned":
					_,logits = model.module.forward_unaligned(x1, phone_target_sequence,  X_bertpad, input_lens, X_bert_len)

				if modalities == "a_phfeat_aligned_unaligned":
					_,logits = model.module.forward_aligned_unaligned(src=x1, txt_aligned=phone_target, txt_unaligned=phone_target_sequence,\
						trans_aligned=X_bert_extendpad , trans_unaligned = X_bertpad, \
						wav_len=input_lens, trans_aligned_len=X_bert_extend_len, trans_unaligned_len = X_bert_len, \
						phone_aligned_len= phone_target_len, phone_unaligned_len = phone_target_sequence_len)

				if modalities == "a_phfeat_aligned_unaligned-2":
					_,logits = model.module.forward_aligned_unaligned2(src=x1, txt_aligned=phone_target, txt_unaligned=phone_target_sequence,\
						trans_aligned=X_bert_extendpad , trans_unaligned = X_bertpad, \
						wav_len=input_lens, trans_aligned_len=X_bert_extend_len, trans_unaligned_len = X_bert_len, \
						phone_aligned_len= phone_target_len, phone_unaligned_len = phone_target_sequence_len)

				loss2_V, loss2_A, loss2_D, attention = None, None, None, None

			pred = torch.max(logits, dim=1)[1]
			correct += pred.eq(target).sum()
			all_predictions.extend(pred.detach().cpu().numpy())
			y_test.extend(target.cpu().numpy())
			total_loop += len(target)
			
			phone_segmented.extend(phone_target)
			emo_target.extend(target)

			# logging.info(logits_vad, vad_target)
			for i in range(len(target)):
				cm[target[i]][pred[i]] += 1

		if not isinstance(correct, int) :
			correct = correct.item()
		perc = correct / total_loop
		logging.info("---- DEV/TEST ----")
		logging.info("Correct: %.2f, Total: %.2f, Perc:  %.2f" %(correct, total_loop, perc ))
		if model_type == "attn":
			logging.info("MSE V: %.3f, A: %.3f, D: %.3f" %(loss2_V, loss2_A, loss2_D))
		logging.info(cm)
		metrics = get_metrics(all_predictions, y_test, "test-")
		log_dictionary=metrics
		log_dictionary['test_acc'] = perc
		log_dictionary['test_cm'] = cm
		wandb.log(log_dictionary)
		return  loss2_V, loss2_A, loss2_D, perc, attention, phone_segmented, emo_target #torch.cat(attention, dim=0).cpu()


def train_epoch_trans(model, train_loader, optimizer):
	model.train()
	running_loss = 0.0
	start_time = time.time()
	c = 0
	correct, total_loop = 0, 0
	attention, phone_segmented, emo_target = [], [] , []
	# vad is a tuple of three = (V, A, D)
	log_dictionary = {}
	all_predictions = []
	y_test = []
	for x1, target, x2, vad_target, X_bertpad, X_bert_len, phone_target, input_lens, \
			phone_target_sequence, X_bert_extendpad, X_bert_extend_len, phone_target_len , phone_target_sequence_len in tqdm(train_loader): # (max_len x bs x dim) # (max_len x bs) #bs #bs
	# for items in tqdm(train_loader):
		if len(x1) < 2:
			continue
		optimizer.zero_grad()
		# target = items[2][1]
		# phone_target = items[2][6]
		# print("Targets: ",target.shape)
		# target = torch.LongTensor(target).cuda() if cuda else torch.LongTensor(target)
		if model_type == "attn":
			logits, logits_vad, _, attns = model(x1, x2)

		if model_type == "5cnn" or model_type == "se_resnet50":
			logits = model(x1)
		if model_type == "transformer":

			# input_lens = torch.Tensor(input_lens).cuda()
			# X_bert_len = torch.Tensor(X_bert_len).cuda()
			# X_bert_extend_len = torch.Tensor(X_bert_extend_len).cuda()
			# phone_target_len = torch.Tensor(phone_target_len).cuda()
			# phone_target_sequence_len = torch.Tensor(phone_target_sequence_len).cuda()
			if modalities == 'a_inputspace':
				_, logits = model(x1,input_lens)
			if modalities=='a_ph_inputspace':
				x1=torch.cat((x1, phone_target),2)
				logits = model(x1,input_lens)
			if modalities == "a_phfeatspace_aligned":
				_, logits = model(src=x1, txt=phone_target, trans=X_bert_extendpad , wav_len=input_lens, bert_len=X_bert_extend_len)
				# model(items)
				
			if modalities == "a_phfeatspace_aligned_2":
				_,logits = model.module.forward2(src=x1, txt=phone_target, trans=X_bert_extendpad , wav_len=input_lens, bert_len=X_bert_extend_len)
			if modalities == "a_phfeat_unaligned":
				_,logits = model.module.forward_unaligned(x1, phone_target_sequence, X_bert_extendpad, input_lens, X_bert_len)
			if modalities == "a_phfeat_aligned_unaligned":
				_,logits = model.module.forward_aligned_unaligned(src=x1, txt_aligned=phone_target, txt_unaligned=phone_target_sequence,\
					trans_aligned=X_bert_extendpad , trans_unaligned = X_bertpad, \
					wav_len=input_lens, trans_aligned_len=X_bert_extend_len, trans_unaligned_len = X_bert_len, \
					phone_aligned_len= phone_target_len, phone_unaligned_len = phone_target_sequence_len)
			if modalities == "a_phfeat_aligned_unaligned-2":
				_,logits = model.module.forward_aligned_unaligned2(src=x1, txt_aligned=phone_target, txt_unaligned=phone_target_sequence,\
					trans_aligned=X_bert_extendpad , trans_unaligned = X_bertpad, \
					wav_len=input_lens, trans_aligned_len=X_bert_extend_len, trans_unaligned_len = X_bert_len, \
					phone_aligned_len= phone_target_len, phone_unaligned_len = phone_target_sequence_len)
			if modalities == "a_phfeat_unaligned_2":#G model
				_,logits = model.module.forward_unaligned_2(x1, phone_target_sequence, X_bert_extendpad, input_lens, X_bert_len)


		pred = torch.max(logits, dim=1)[1]
		all_predictions.extend(pred.detach().cpu().numpy())
		y_test.extend(target.cpu().numpy())
		correct += pred.eq(target).sum()
		total_loop += len(target)

		loss1 = F.nll_loss(logits, target, weight=class_balancing_weights)
		
		if model_type == "attn":
			loss2_V = F.mse_loss(logits_vad[:,0], vad_target[:,0])
			loss2_A = F.mse_loss(logits_vad[:,1], vad_target[:,1])
			loss2_D = F.mse_loss(logits_vad[:,2], vad_target[:,2])
			total_loss = loss1 + loss2_V + loss2_A + loss2_D#. Not using the VAD but only the cross entropy loss
			attention.append(attns)

		if model_type == "5cnn" or model_type == "se_resnet50" or model_type == 'transformer':
			total_loss = loss1
			loss2_V, loss2_A, loss2_D, attention = None , None, None, None

		total_loss.backward()
		optimizer.step()
		running_loss += (total_loss.item())
		c+=1
		# logging.info(type(attns[0][0]))
		phone_segmented.extend(phone_target) 
		emo_target.extend(target)
		log_dictionary['loss']=total_loss
		wandb.log(log_dictionary)
		# if (c>3):
			# break
	end_time = time.time()
	running_loss /= len(train_loader)
	logging.info("---- TRAIN ----")
	logging.info(correct)
	if not isinstance(correct, int) :
		correct = correct.item()
	running_acc = correct / total_loop
	logging.info("class_loss {%.3f}, total_loss {%.3f}, Time: {%.3f} s" %(loss1, total_loss, end_time - start_time))
	if model_type == "attn":
		logging.info("MSE V: %.3f, A: %.3f, D: %.3f" %(loss2_V, loss2_A, loss2_D))
	logging.info('Training  class_acc: %.3f' % running_acc)
	metrics = get_metrics(all_predictions, y_test)
	log_dictionary=metrics
	log_dictionary['train_acc'] = running_acc
	wandb.log(log_dictionary)
	return model, loss1, loss2_V, loss2_A, loss2_D, running_acc, attention, phone_segmented, emo_target


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--batchsize', metavar='N', type=int, default=8)
	parser.add_argument('--epochs', metavar='E', type=int, default=50)
	parser.add_argument('--modalities', default='a_ph_inputspace', choices=['a_ph_inputspace', 'a_inputspace',\
															 'a_phfeatspace_aligned', 'a_phfeatspace_aligned_2', 'a_phfeat_unaligned',\
															 'a_phfeat_aligned_unaligned', 'a_phfeat_aligned_unaligned-2' , 'a_phfeat_unaligned_2'])
	parser.add_argument('--model_type', default='transformer')
	parser.add_argument('--parallel', action='store_true')
	parser.add_argument('--encoder_module', default=None)
	parser.add_argument('--logfile', default='del')
	parser.add_argument('--learning_rate',  type=float, default=0.001)
	parser.add_argument('--momentum',  type=float, default=0.8)
	# parser.add_argument('--decay',  type=float, default=1e-6)
	parser.add_argument('--optimizer', default='sgd')
	parser.add_argument('--context', default='yes')
	parser.add_argument('--test_speaker', type=int, default=0)
	parser.add_argument('--dataset', type=str, default='IEMOCAP', choices=['IEMOCAP', 'CMU_MOSI'])

	args = parser.parse_args()


	BATCH_SIZE = args.batchsize
	n_epochs = args.epochs
	model_type = args.model_type
	modalities = args.modalities
	MODEL_NAME = args.logfile# "transformer_bs"+str(BATCH_SIZE)+"_"+modalities#+phoneme_class
	PARALLEL = args.parallel
	encoder_module = args.encoder_module
	logfile = args.logfile
	lr = args.learning_rate
	optimizer = args.optimizer
	momentum = args.momentum
	test_speaker = args.test_speaker
	context = args.context
	dataset = args.dataset
	# decay = args.decay

	############### TESTING ###########################
	# if dataset == 'CMU_MOSI':
	# 	loader = CMU_MOSI()
	# 	trainX, trainY, trainTY, trainVAD, trainPhone, trainBert, trainBertExtend = loader.train()
	# 	devX, devY, devTY, devVAD, devPhone, devBert, devBertExtend = loader.test()
	# 	train = Dataset(trainX, trainY, trainTY, trainVAD, trainPhone, trainBert, trainBertExtend, dataset='CMU_MOSI')
	# 	dev = Dataset(devX, devY, devTY, devVAD, devPhone, devBert, devBertExtend, dataset='CMU_MOSI')
	# 	# print(train[0])
	# 	# print(dev[0])
	# 	collate_lines([train[0], train[1]])
	# exit()
	############### TESTING ###########################

	####################### WANDB #####################

	wandb.init(project="my-test-project", entity="ydhira")
	wandb.init(config=args)
	logging.info(args)
	print(args)
	####################### WANDB #####################
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)
	# logfile="logs/"+MODEL_NAME+'-'+encoder_module
	logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode='w',
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

	print("Logging into: " + logfile)

	
	idx = np.array([0,1,2,3,4,5,6,7,8,9])
	train_loss_l, train_acc_l = [], []
	dev_acc_l, test_acc_l = [], [] 
	dev_loss2_Vl, dev_loss2_Al, dev_loss2_Dl, = [], [], []
	test_loss2_Vl, test_loss2_Al, test_loss2_Dl, = [], [], []

	#for j in range(len(idx))[:1]: # only running it for one of the speakers 
	# for j in range(1,2):	#Sweep 1
	# for j in range(2,3): # sweep 2
	for j in range(1): #generic running because now im paasing test speaker as part of args 
		logging.info("J: %d", j )
		# test_idx = [idx[j-1]]
		# dev_idx = [idx[j-1]]
		test_idx = [test_speaker]
		dev_idx = [test_speaker]
		train_idx = np.setdiff1d(idx, test_idx + dev_idx)

		if dataset == 'IEMOCAP':
			loader = IEMOCAP(train_idx, test_idx, dev_idx)
			trainX, trainY, trainTY, trainVAD, trainPhone, trainBert, trainBertExtend, trainorder = loader.train()
			devX, devY, devTY, devVAD, devPhone, devBert, devBertExtend, devorder = loader.dev()
			devTY = list(map(lambda sentence: (' ').join(sentence).replace("\n", "").lower().translate(str.maketrans('','',string.punctuation)) , devTY))
			trainTY = list(map(lambda sentence: (' ').join(sentence).replace("\n", "").lower().translate(str.maketrans('','',string.punctuation)) , trainTY))

			dev_vocab =  list(set(np.concatenate(list(map(lambda x: x.split(' '), devTY)))))
			train_vocab = list(set(np.concatenate(list(map(lambda x: x.split(' '), trainTY)))))
			target_vocab = list(set(np.concatenate((dev_vocab, train_vocab))))
			classn = len(set(trainY))
			unique, counts = np.unique(trainY, return_counts=True)
			# logging.info((' ').join(unique), (' ').join(counts))
			logging.info(counts/sum(counts))
			class_balancing_weights = counts/sum(counts)
			
			class_balancing_weights = (1 - (counts/sum(counts))) / (1 - (counts/sum(counts))).sum()
			class_balancing_weights_corr = []
			for idx_uni in unique:
				class_balancing_weights_corr.append(\
					class_balancing_weights[emotions_used.index(idx_uni)])
			class_balancing_weights = torch.tensor(class_balancing_weights_corr).cuda().float() if cuda \
									else torch.tensor(class_balancing_weights_corr).float()
			logging.info(class_balancing_weights)
			logging.info("No of classes %d " %(classn))

		if dataset == 'CMU_MOSI':
			loader = CMU_MOSI()
			trainX, trainY, trainTY, trainVAD, trainPhone, trainBert, trainBertExtend = loader.train()
			devX, devY, devTY, devVAD, devPhone, devBert, devBertExtend = loader.test()
			classn=2
			class_balancing_weights=[0.5,0.5]
			class_balancing_weights = torch.tensor(class_balancing_weights).cuda().float() 
		# logging.info(str(trainY[0]) + " " + str(trainVAD[0]))


		if model_type == "attn":
			model = self_attn_model( 1, 4, 16, 64, 4, target_vocab)
		if model_type == "5cnn":
			model = Net(1, 4, 16, 64, 256, class_n=classn)
		if model_type == "se_resnet50":
			model = se_resnet50(num_classes=classn)
		if model_type == "transformer":
			if modalities=='a_ph_inputspace':
				model = TransformerClassAudio(
        			#tgt_vocab, input_size, d_model, d_pos_enc, nhead, num_encoder_layer, num_decoder_layer, d_ffn, n_class
		        	720, 1024, 512, 512,  8, 1, 1, 1024, classn,  activation=torch.nn.GELU, encoder_module=encoder_module, normalize_before=True
		    		)
			if modalities == "a_inputspace":
				model = TransformerClassAudio(
		        		#tgt_vocab, input_size, d_model,d_pos_enc, nhead, num_encoder_layer, num_decoder_layer, d_ffn, n_class
		        		720, 1024, 64, 64, 8, 1, 1, 64, classn, activation=torch.nn.GELU, encoder_module=encoder_module, normalize_before=True
		    		)
			if modalities == "a_phfeatspace_aligned":
				model = TransformerClassAudioText(
		        		#tgt_vocab, input_size, d_model, d_pos_enc, nhead, num_encoder_layer, num_decoder_layer, d_ffn, n_class
		        		720, 1024, 128, 128, 8, 1, 1, 128, classn, modalities=modalities, activation=torch.nn.GELU, encoder_module=encoder_module, normalize_before=True
		    		)
			# if modalities == "a_phfeatspace_aligned_2":
			# 	model = TransformerClassAudioText(
		 #        		#tgt_vocab, input_size, d_model, d_pos_enc, nhead, num_encoder_layer, num_decoder_layer, d_ffn, n_class
		 #        		720, 1024, 64, 64*3, 8, 1, 1, 64, classn, modalities=modalities, activation=torch.nn.GELU, encoder_module=encoder_module, normalize_before=True
		 #    		)
			if modalities == "a_phfeat_unaligned":
				model = TransformerClassAudioText(
		        		#tgt_vocab, input_size, d_model, d_pos_enc, nhead, num_encoder_layer, num_decoder_layer, d_ffn, n_class
		        		720, 1024, 128, 128, 8, 1, 1, 128, classn, modalities=modalities, activation=torch.nn.GELU, encoder_module=encoder_module, normalize_before=True
		    		)
			if modalities == "a_phfeat_unaligned_2":
				model = TransformerClassAudioText(
		        		#tgt_vocab, input_size, d_model, d_pos_enc, nhead, num_encoder_layer, num_decoder_layer, d_ffn, n_class
		        		720, 1024, 64, 64, 8, 1, 1, 64, classn, modalities=modalities, activation=torch.nn.GELU, encoder_module=encoder_module, normalize_before=True
		    		)
			if modalities == "a_phfeat_aligned_unaligned":
				model = TransformerClassAudioText(
		        		#tgt_vocab, input_size, d_model, d_pos_enc, nhead, num_encoder_layer, num_decoder_layer, d_ffn, n_class
		        		720, 1024, 128, 128, 8, 1, 1, 128, classn, modalities=modalities, activation=torch.nn.GELU, encoder_module=encoder_module, normalize_before=True
		    		)
			if modalities == "a_phfeat_aligned_unaligned-2":
				model = TransformerClassAudioText(
		        		#tgt_vocab, input_size, d_model, d_pos_enc, nhead, num_encoder_layer, num_decoder_layer, d_ffn, n_class
		        		720, 1024, 64, 64*3, 8, 1, 1, 64, classn, modalities=modalities, activation=torch.nn.GELU, encoder_module=encoder_module, normalize_before=True
		    		)
			

		if PARALLEL:
			model = nn.DataParallel(model, device_ids=[0,1])

		# device = torch.device("cuda" if cuda else "cpu")
		# print(device)
		model = model.cuda() #if cuda else model.to(device)

		
		if context == 'no':
			train = Dataset(trainX, trainY, trainTY, trainVAD, trainPhone, trainBert, trainBertExtend, dataset=dataset)
			dev = Dataset(devX, devY, devTY, devVAD, devPhone, devBert, devBertExtend, dataset=dataset)
			dataloader_args = dict(shuffle=True, batch_size=BATCH_SIZE, collate_fn = collate_lines, num_workers=0) 

			train_loader = dataloader.DataLoader(train, **dataloader_args)
			dataloader_args = dict(shuffle=False, batch_size=BATCH_SIZE, collate_fn = collate_lines, num_workers=0)
			dev_loader = dataloader.DataLoader(dev, **dataloader_args)

		if context == 'yes':
			train = DatasetContext(trainX, trainY, trainTY, trainVAD, trainPhone, trainBert, trainBertExtend, trainorder, dataset=dataset)
			dev = DatasetContext(devX, devY, devTY, devVAD, devPhone, devBert, devBertExtend, devorder, dataset=dataset)
			dataloader_args = dict(shuffle=True, batch_size=BATCH_SIZE, collate_fn = collate_lines_context, num_workers=0) 

			train_loader = dataloader.DataLoader(train, **dataloader_args)
			dataloader_args = dict(shuffle=False, batch_size=BATCH_SIZE, collate_fn = collate_lines_context, num_workers=0)
			dev_loader = dataloader.DataLoader(dev, **dataloader_args)
		# test = Dataset(testX, testY, testTY, testVAD, testPhone, testBert, test=True)

		logging.info("Train shape %s, Dev shape %s " %(trainX.shape, devX.shape))
		logging.info("Testing for speaker: %d " %(dev_idx[0]))



		# test_loader = dataloader.DataLoader(test, **dataloader_args)

		if optimizer == "sgd":
			optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
		if optimizer == "adam":
			optimizer = optim.Adam(model.parameters(), lr=lr)

		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, threshold=1e-1, threshold_mode='abs')
		

		for i in range(n_epochs):
			logging.info("Starting Training Epoch %d " % i)
			# dev_loss2_V, dev_loss2_A, dev_loss2_D, dev_perc, dev_attention, dev_phone_segmented, dev_emo_target = val_model_trans(model, dev_loader)
			model, train_loss1, train_loss2_V, train_loss2_A, train_loss2_D, train_acc, train_attention, train_phone_segmented, train_emo_target \
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

			fname = "models/"+str(test_speaker)+"_"+str(i)+MODEL_NAME+".model"
			print("Saving to: ", fname)
			torch.save({'epoch': i, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': dev_loss2_V, 'acc': dev_perc}, fname)


	logging.info(train_loss_l)
	logging.info(train_acc_l)
	logging.info(dev_acc_l)
	logging.info(dev_loss2_Vl)
	logging.info(dev_loss2_Al)
	logging.info(dev_loss2_Dl)