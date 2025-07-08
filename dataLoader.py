'''
DataLoader for training
This model is modified based on the following project:
  1. https://github.com/TaoRuijie/ECAPA-TDNN
'''
import glob, numpy, os, random, soundfile, torch
from scipy import signal

class train_loader(object):
	def __init__(self, args, data_list, data_label, musan_path, rir_path, num_frames, labelled=True):
		self.args = args
		self.data_list = data_list
		self.data_label= data_label
		self.num_frames = num_frames
		self.labelled=labelled

		# Load and configure augmentation files
		self.noisetypes = ['noise','speech','music'] # Type of noise
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]} # The range of SNR
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		self.noiselist = {}
		augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
		for file in augment_files:
			if file.split('/')[-4] not in self.noiselist:
				self.noiselist[file.split('/')[-4]] = []
			self.noiselist[file.split('/')[-4]].append(file)
		self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))# Load the rir file

	def __getitem__(self, index):
		audio = loadWAV(self.data_list[index], self.num_frames).astype(numpy.float64)
		# Data Augmentation
		if self.labelled:
			augtype = random.randint(0, 5)
			aug_audio = self.augment_wav(audio, augtype)
			return index, torch.FloatTensor(aug_audio[0]), self.data_label[index]
		else:
			augtype = random.randint(1, 5)
			aug_audio = self.augment_wav(audio, augtype)
			return index, torch.FloatTensor(audio[0]), torch.FloatTensor(aug_audio[0]), self.data_label[index]

	def __len__(self):
		return len(self.data_list)

	def augment_wav(self, audio, augtype):
		if augtype == 0:   # Original
			audio = audio
		elif augtype == 1:  # Reverberation
			audio = self.add_rev(audio)
		elif augtype == 2:  # Babble
			audio = self.add_noise(audio, 'speech')
		elif augtype == 3:  # Music
			audio = self.add_noise(audio, 'music')
		elif augtype == 4:  # Noise
			audio = self.add_noise(audio, 'noise')
		elif augtype == 5:  # Television noise
			audio = self.add_noise(audio, 'speech')
			audio = self.add_noise(audio, 'music')
		return audio

	def add_rev(self, audio):
		rir_file    = random.choice(self.rir_files)
		rir, sr     = soundfile.read(rir_file)
		rir         = numpy.expand_dims(rir.astype(numpy.float64),0)
		rir         = rir / numpy.sqrt(numpy.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:,:self.num_frames * 160 + 240]

	def add_noise(self, audio, noisecat):
		clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 
		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
		noises = []
		for noise in noiselist:
			noiseaudio, sr = soundfile.read(noise)
			length = self.num_frames * 160 + 240 # Length of segment for training
			if noiseaudio.shape[0] <= length:
				shortage = length - noiseaudio.shape[0]
				noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
			else:
				start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))# If length is enough
				noiseaudio = noiseaudio[start_frame:start_frame + length]# Only read some part to improve speed
			noiseaudio = numpy.stack([noiseaudio],axis=0)
			noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio

def loadWAV(filename, max_frames):
	# Read the utterance and randomly select the segment
	audio, sr = soundfile.read(filename)
	# Length of segment for training
	length = max_frames * 160 + 240  # 240 is for padding, for 15ms since window is 25ms and step is 10ms.
	if audio.shape[0] <= length:  # Padding if less than required length
		shortage = length - audio.shape[0]
		audio = numpy.pad(audio, (0, shortage), 'wrap')
	start_frame = numpy.int64(random.random() * (audio.shape[0] - length))  # Randomly select a start frame to extract audio
	audio = numpy.stack([audio[start_frame:start_frame + length]], axis=0)
	return audio

