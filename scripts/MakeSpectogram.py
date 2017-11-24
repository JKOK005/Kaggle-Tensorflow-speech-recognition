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

	def getFreqRange(self, smpl_interval_size):
		freq_range = np.linspace(0, self.framerate, smpl_interval_size *2)
		return freq_range[:smpl_interval_size]

	def getTimeRange(self, total_smpl_size):
		time_range = np.array([i*self.smpl_interval for i in range(total_smpl_size)])
		time_range_offset = time_range + self.smpl_interval
		return time_range_offset

	def mapToFFT(self, arr):
		fft_arr 		= sc.fft(arr)
		valid_arr_range = len(arr) / 2
		return map(abs, fft_arr[:valid_arr_range])

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

	# Print frequency band and time series range
	print(spectogram.getFreqRange(len(spectogram_arr[0])))
	print(spectogram.getTimeRange(len(spectogram_arr)))

	import matplotlib.pyplot as plt 
	import seaborn as sns

	y_time = spectogram.getTimeRange(len(spectogram_arr))
	x_freq = spectogram.getFreqRange(len(spectogram_arr[0]))

	sample_spectogram = np.array(spectogram_arr[8])
	spectogram_to_nparr = np.array(spectogram_arr)
	fig, ax = plt.subplots(2,1)
	# Visualization of single fft band for Nth interval
	ax[0].plot(x_freq, sample_spectogram)
	ax[0].set_xlabel('Frequency band (Hz)')
	ax[0].set_ylabel('FFT response') 	
	# Visualization of spectogram via heatmap
	sns.heatmap(spectogram_to_nparr, ax=ax[1], cmap='RdBu_r')
	ax[1].set_xlabel('Frequency band (bins)')
	ax[1].set_ylabel('Sample count')
	plt.show()

