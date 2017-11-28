import tensorflow as tf 
from ModelInterface import ModelInterface
from tensorflow.contrib import rnn

class SingleLayerLSTMNetwork(ModelInterface):
		timesteps 		= None
		num_input 		= None
		num_output 		= None
		hidden_units 	= None
		loss 			= None
		optimizer 		= None
		dropout 		= None
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

		def setDropout(self, dropout):
			self.dropout = dropout
			return self

		def build(self):
			network = rnn.LSTMCell(self.hidden_units)
			if(self.dropout is not None):
				network = rnn.DropoutWrapper(network, output_keep_prob=self.dropout)

			input_tensor 	= tf.placeholder(tf.float32, [None, None, self.num_input])
			output, _ 	 	= tf.nn.dynamic_rnn(network, input_tensor, dtype=tf.float32)
			last_output  	= tf.transpose(output, [1,0,2])
			last 			= tf.gather(last_output, int(last_output.get_shape()[0]) - 1)
			import IPython
			IPython.embed()

		def getModel(self):
			pass

if __name__ == "__main__":
	lstm = SingleLayerLSTMNetwork()
	lstm.setHiddenUnits(100).setInputSize(160)
	lstm.build()