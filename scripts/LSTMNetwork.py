import tensorflow as tf 
from ModelInterface import ModelInterface

class LSTMNetwork(ModelInterface):
		seq_units 		= None
		input_size 		= None
		output_size 	= None
		hidden_units 	= None
		loss 			= None
		optimizer 		= None

		def __init__(self):
			pass

		def setNoOfUnits(self, no_seq_units):
			self.seq_units = no_seq_units
			return self

		def setInputSize(self, input_size):
			self.input_size = input_size
			return self

		def setOutputSize(self, output_size):
			self.output_size = output_size
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

		def build(self):
			pass

		def getModel(self):
			pass

if __name__ == "__main__":
	lstm = LSTMNetwork()
	lstm.build()