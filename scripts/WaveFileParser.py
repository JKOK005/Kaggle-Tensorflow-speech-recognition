import wave

class WaveFileParser(object):
	
	def __hexStrToStrArr(self, str, delimiter):
		pass

	@classmethod
	def readWaveAsHexString(cls, wav_file):
		# getnframes -> Gets the number of sample record for the voice
		return wave.open(wav_file, 'rb')

	@classmethod
	def hexStringToIntArr(cls, str):
		pass