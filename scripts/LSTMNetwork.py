import tensorflow as tf 
from ModelInterface import ModelInterface
from tensorflow.contrib import rnn

class SingleLayerLSTMNetwork(ModelInterface):
		timesteps 		= None
		num_input 		= None
		num_output 		= None
		hidden_units 	= None
		cost 			= None
		optimizer 		= None
		dropout 		= None
		learning_rate 	= None

		__prediction 	= None
		__objective 	= None
		__error 		= None

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

		def setLossFunction(self, loss_type):
			self.cost = loss_type
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

		def __getLoss(self, loss_type):
			loss = None
			output_tensor = tf.placeholder(tf.float32, [None, self.num_output]) 

			if(loss_type == 'cross_entropy'):
				loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.__prediction, labels=output_tensor))
			return loss

		def __getOptimizer(self, optimizer_type, learning_rate):
			optimizer = None
			if(optimizer_type == 'gradient_descent'):
				optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			return optimizer

		def __setObjective(self):
			loss = self.__getLoss(self.cost)
			optimizer = self.__getOptimizer(self.optimizer, self.learning_rate)
			self.__objective = optimizer.minimize(loss)
			return

		def build(self):
			network = rnn.LSTMCell(self.hidden_units)
			if(self.dropout is not None):
				network = rnn.DropoutWrapper(network, output_keep_prob=self.dropout)

			input_tensor 	= tf.placeholder(tf.float32, [None, self.timesteps, self.num_input])

			output, _ 	 	= tf.nn.dynamic_rnn(network, input_tensor, dtype=tf.float32)
			transpose_time_axis = tf.transpose(output, [1,0,2])
			last_timestamp 	= tf.gather(transpose_time_axis, int(transpose_time_axis.get_shape()[0]) - 1)

			init_weight 	= tf.Variable(tf.random_normal([self.hidden_units, self.num_output]))
			init_bias 		= tf.Variable(tf.random_normal([self.num_output]))

			self.__prediction = tf.nn.softmax(tf.matmul(last_timestamp, init_weight) + init_bias)
			self.__setObjective()
			
		def getModel(self):
			pass

if __name__ == "__main__":
	lstm = SingleLayerLSTMNetwork()
	lstm.setHiddenUnits(100).setInputSize(160).setOutputSize(12).setTimeSteps(40) \
		.setLossFunction('cross_entropy').setOptimizer('gradient_descent').setLearningRate(0.01) \
		.build()