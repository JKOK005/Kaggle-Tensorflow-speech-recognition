import numpy as np
import scipy as sc
from WaveFileParser import WaveFileParser

class MakeSpectogram(object):
	parser 			= WaveFileParser()
	smpl_interval 	= None 	# Seconds
	wav_file 		= None
	framerate 		= None
	nframes 		= None

	def __init__(self, wav_file):
		self.wav_file 	= wav_file
		(nchannels, sampwidth, framerate, nframes, comptype, compname) = WaveFileParser.getMetaData(self.wav_file)
		self.framerate 	= framerate
		self.nframes 	= nframes

	def setSmplInterval(self, interval):
		self.smpl_interval 	= interval

	def getFreqRange(self, smpl_len):
		permissible_freq 	= self.framerate / 2
		half_smpl_len 		= int(smpl_len / 2)
		return np.linspace(0, permissible_freq, half_smpl_len)

	def mapToFFT(self, arr):
		fft_arr 	= sc.fft(arr)
		return map(abs, fft_arr)

	def create(self):
		period 			= 1.0 / self.framerate
		smpl_size 		= int(self.smpl_interval / period)
		spectogram_arr 	= []
		for each_interval in np.arange(smpl_size, self.nframes, smpl_size):
			frame_start = each_interval -smpl_size
			frame_end 	= each_interval -1
			int_arr 	= WaveFileParser.parseWavFile(self.wav_file, frame_start=frame_start, frame_end=frame_end)
			fft_arr 	= self.mapToFFT(int_arr)
			spectogram_arr.append(fft_arr)
		return spectogram_arr

if __name__ == "__main__":
	spectogram = MakeSpectogram("test.wav")
	spectogram.setSmplInterval(0.020)
	spectogram_arr = spectogram.create()
	print(spectogram.getFreqRange(len(spectogram_arr[0])))

	# Visualization 
	import matplotlib.pyplot as plt 
	sample_spectogram = np.array(spectogram_arr[10])
	plt.plot(sample_spectogram)
	plt.show()