## This wrapper is using the file Interspeech 2022 transfomer cadence work 
from transformer_class import TransformerClassAudio
import argparse
import torch 
import torch.nn as nn 
from transformers import Wav2Vec2Processor, HubertModel
import soundfile as sf
import numpy as np 

MyHubertProcessor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
MyHubertModel = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
emotions_used = ['Anger', 'Happy','Sad', 'Neutral']

def get_feature(audio_file):
	# fill this in
	audio, sr = sf.read(audio_file)
	if len(audio.shape)==2: # some files are 2 channels, some are one 
		audio = np.mean(audio, axis=1)
	input_values = MyHubertProcessor(audio, sampling_rate=sr, return_tensors="pt").input_values  # Batch size 1
	hidden_states = MyHubertModel(input_values).last_hidden_state

	hidden_states=hidden_states.squeeze().detach().numpy()
	return hidden_states



## Define a model class 
def run_main(model, device, audio_file):
	print("processing file: %s" %audio_file )
	feature_input = get_feature(audio_file).astype(np.float32)
	feature_input_expand = torch.Tensor(np.expand_dims(feature_input, 0)).to(device)
	input_lens = torch.Tensor([feature_input_expand.shape[1]]).to(device).reshape(-1)

	with torch.no_grad():
		_, logits =  model(feature_input_expand, input_lens)
	pred = torch.max(logits, dim=1)[1]

	return pred 


if __name__ == '__main__':
	## Get a file 
	parser = argparse.ArgumentParser( prog='vad_wrapper.py', description='Takes in the input wav file and outputs \
																	(1) predicted emotion class ')

	parser.add_argument('-audio_file', required=True, help='The input audio file to process')
	args = parser.parse_args()

	audio_file = args.audio_file
	cuda = True 
	classn = 4
	encoder_module = 'conformer'
	print("Processing audio file: %s ..." %(audio_file))
	device = torch.device("cuda:0" if cuda else "cpu")
	model = TransformerClassAudio(
	    		#tgt_vocab, input_size, d_model,d_pos_enc, nhead, num_encoder_layer, num_decoder_layer, d_ffn, n_class
	    		720, 1024, 64, 64, 8, 1, 1, 64, classn, activation=torch.nn.GELU, encoder_module=encoder_module, normalize_before=True)
	modelpath='/home/hyd/workhorse2/multi-hop-attention/code-clean/transformer/models/2_49transformer-bs6-a_inputspace-conformer_adam_hubert_iemocap.model'
	device = torch.device("cuda:0" if cuda else "cpu")
	model = model.to(device).cuda() if cuda else model.to(device)
	model = nn.DataParallel(model, device_ids=[0,1])
	checkpoint = torch.load(modelpath)
	model.load_state_dict(checkpoint['model_state_dict'])

	model.eval()

	pred_emotion = run_main(model, device, audio_file)
	print("Predicted emotion: %s" %(emotions_used[pred_emotion]))
	print('Done :)')

