import tensorflow as tf 
from ModelInterface import ModelInterface
from tensorflow.contrib import rnn

class LSTMNetwork(ModelInterface):
		timesteps 		= None
		num_input 		= None
		num_output 		= None
		hidden_units 	= None
		loss 			= None
		optimizer 		= None
		learning_rate 	= None

		def __init__(self):
			pass

		def setTimeSteps(self, timesteps):
			self.timesteps = timesteps
			return self

		def setInputSize(self, input_size):
			self.num_input = input_size
			return self

		def setOutputSize(self, output_size):
			self.num_output = output_size
			return self

		def setHiddenUnits(self, no_hidden_units):
			self.hidden_units = no_hidden_units
			return self

		def setLossFunction(self, loss_fn):
			self.loss = loss_fn
			return self

		def setOptimizer(self, optimizer):
			self.optimizer = optimizer
			return self

		def setLearningRate(self, learning_rate):
			self.learning_rate = learning_rate
			return self

		def build(self):
			pass

		def getModel(self):
			pass

if __name__ == "__main__":
	lstm = LSTMNetwork()
	lstm.build()