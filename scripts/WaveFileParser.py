import wave
import struct

class WaveFileParser(object):
	@classmethod
	def __readWaveAsHexString(self, wav_file):
		# getnframes -> Gets the number of sample record for the voice
		return wave.open(wav_file, 'rb')

	@classmethod
	def __hexStringToIntArr(self, wav_obj):
		frame_one 	= wav_obj.readframes(1)
		int_arr 	= []
		while(frame_one is not ''):
			hex_to_val = struct.unpack('h', frame_one)
			hex_to_val_unpack = hex_to_val[0]
			int_arr.append(hex_to_val_unpack)
			frame_one = wav_obj.readframes(1)
		return int_arr

	@classmethod
	def parseWavFile(cls, wav_file):
		wave_obj = cls.__readWaveAsHexString(wav_file)
		return cls.__hexStringToIntArr(wave_obj)

if __name__ == "__main__":
	print(WaveFileParser.parseWavFile("test.wav"))