import subprocess, os

IN_DIR="/home/hyd/workhorse2/multi-hop-attention/data/IEMOCAP_full_release/"

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

zero_speech_dir = "/home/hyd/workhorse2/deepfake/ZeroSpeech/"
wav_files = get_req_files(IN_DIR, ".wav", "wav/")
for wav_file in wav_files:
	print(wav_file)
	res=subprocess.run(['python', zero_speech_dir+'convert.py', 'checkpoint='+zero_speech_dir+'checkpoints/2019english/model-resuming-.ckpt-380000.pt', \
		'in_dir=inwavs', 'out_dir=outwavs', 'dataset=2019/english'])
	print(res)
	exit()