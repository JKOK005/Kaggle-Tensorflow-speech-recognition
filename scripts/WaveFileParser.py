import wave
import struct

class WaveFileParser(object):
	@classmethod
	def __openWavFile(cls, wav_file, access):
		# getnframes -> Gets the number of sample record for the voice
		return wave.open(wav_file, access)

	@classmethod
	def __setWavPointer(cls, wav_obj, frame_start):
		limit = wav_obj.getnframes()
		if(frame_start > limit):
			raise Exception("Frame position exceeds allowable frames")
		wav_obj.setpos(frame_start)
		return wav_obj

	@classmethod
	def __wavToIntArrWithLen(cls, wav_obj, length):
		int_arr = []
		for itr in range(length):
			one_frame = wav_obj.readframes(1)
			hex_to_val = struct.unpack('h', one_frame)
			hex_to_val_unpack = hex_to_val[0]
			int_arr.append(hex_to_val_unpack)
		return int_arr

	@classmethod
	def parseWavFile(cls, wav_file, frame_start=0, frame_end=None):
		wav_obj 	= cls.__openWavFile(wav_file, 'rb')
		wav_obj 	= cls.__setWavPointer(wav_obj, frame_start)
		
		if(frame_end is None):
			frame_end = wav_obj.getnframes()
		return cls.__wavToIntArrWithLen(wav_obj, frame_end -frame_start +1)

if __name__ == "__main__":
	import matplotlib.pyplot as plt 
	import numpy as np
	wav_as_list = WaveFileParser.parseWavFile("test.wav", 10001, 10000)
	wav_as_np_arr = np.array(wav_as_list)
	plt.plot(wav_as_np_arr)
	plt.show()
