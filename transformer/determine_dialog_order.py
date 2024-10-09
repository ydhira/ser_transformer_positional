import glob 
import os 

for sesnum, ses in enumerate(['Session1', 'Session2', 'Session3', 'Session4', 'Session5']):
	sesnum=sesnum+1
	ses_folder = '/share/workhorse2/hyd/multi-hop-attention/data/IEMOCAP_full_release/'+ses+'/sentences/wav/'
	datafolders = \
		os.listdir(ses_folder)# Ses01F_impro01/' # Session1 only, one combination only 

	datafolders = list(map(lambda x: ses_folder + x, datafolders))

	for datafolder in datafolders:
		sessiondialog = datafolder.split('/')[-1]
		print(datafolder)
		spk1_files=glob.glob(datafolder + "/Ses0"+str(sesnum)+"*F0*.wav")
		spk2_files=glob.glob(datafolder + "/Ses0"+str(sesnum)+"*M0*.wav")
		spk1_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0][1:]))
		spk2_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0][1:]))
		# print(spk1_files)
		# print(spk2_files)

		# print(len(spk1_files), len(spk2_files))

		order=[]

		for j in range(min(len(spk1_files), len(spk2_files))):
			order.append(spk1_files[j])
			order.append(spk2_files[j])
		if j < len(spk1_files):
			order.extend(spk1_files[j+1:])
		if j < len(spk2_files):
			order.extend(spk2_files[j+1:])

		tofile = 'dialoge_order/'+ses+'-'+sessiondialog+'.txt'
		print(tofile)
		with open(tofile, 'w') as f:
			for o in order:
				f.write(o+"\n")
		f.close()

