from transformer_class import TransformerClassAudio, TransformerClassAudioText


def run_main(file):
	print("processing file: %s" %file )
	feature_input = get_feature(file).astype(np.float32)
	# feature_input_expand = np.expand_dims(feature_input, 1)

	with torch.no_grad():
		logits, logits_vad =  model(feature_input_expand, embeds)
		pred = torch.max(logits, dim=1)[1]


	modalities = 'a_inputspace'
	if modalities == "a_inputspace":
		
	model = model.cuda()
	if modalities == 'a_inputspace':
		_, logits = model(x1,input_lens)
		pred = torch.max(logits, dim=1)[1]

if __name__ == '__main__':

	parser = argparse.ArgumentParser(prog='gcloud_main.py', description='Takes in the input wav file and outputs \
																	(1) a list of phonemes and (2) time stamps for each phoneme')
	parser.add_argument('-infile', required=True, help='The input file to process')
	args = parser.parse_args()



	model = multi_attention_model(768, 1, 4, target_vocab) # adding back the neutral class
	device = torch.device("cuda:0" if cuda else "cpu")
	model = TransformerClassAudio(
	    		#tgt_vocab, input_size, d_model,d_pos_enc, nhead, num_encoder_layer, num_decoder_layer, d_ffn, n_class
	    		720, 1024, 64, 64, 8, 1, 1, 64, classn, activation=torch.nn.GELU, encoder_module=encoder_module, normalize_before=True)
	checkpoint = torch.load('/home/hyd/multi-hop-attention/code/models/0_14.model')
	model.load_state_dict(checkpoint['model_state_dict'])
	# emotions_used = ['ang', 'hap','sad', 'neu']

	model.eval()
	emo = run_main(args.infile)
	print("Predicted emotion: %s, V: %.2f, A: %.2f, D: %.2f" %(emotions_used[emo[0]], v, a, d))