import tensorflow as tf 
from ModelInterface import ModelInterface
from tensorflow.contrib import rnn
from LossUtilCOR import LossCORHeader
from OptimizerUtilCOR import OptimizerCORHeader

class SingleLayerLSTMNetwork(ModelInterface):
		timesteps 		= None
		num_input 		= None
		num_output 		= None
		hidden_units 	= None
		loss_type 		= None
		optimizer 		= None
		dropout 		= None
		learning_rate 	= None
		
		__input_tensor 	= None
		__output_tensor = None

		__prediction 	= None
		__loss 			= None
		__objective 	= None

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
			self.loss_type = loss_type
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
			loss_fn = LossCORHeader.get(loss_type)
			return tf.reduce_mean(loss_fn(logits=self.__prediction, labels=self.__output_tensor))

		def __getOptimizer(self, optimizer_type, learning_rate):
			optimizer_fn 	= OptimizerCORHeader.get(optimizer_type)			
			optimizer 		= optimizer_fn(learning_rate=learning_rate)
			return optimizer

		def __setObjective(self):
			self.__loss = self.__getLoss(self.loss_type)
			optimizer = self.__getOptimizer(self.optimizer, self.learning_rate)
			self.__objective = optimizer.minimize(self.__loss)
			return

		def build(self):
			network = rnn.LSTMCell(self.hidden_units)
			if(self.dropout is not None):
				network = rnn.DropoutWrapper(network, output_keep_prob=self.dropout)

			self.__input_tensor = tf.placeholder(tf.float32, [None, self.timesteps, self.num_input])
			self.__output_tensor = tf.placeholder(tf.float32, [None, self.num_output]) 

			output, _ 	 	= tf.nn.dynamic_rnn(network, self.__input_tensor, dtype=tf.float32)
			transpose_time_axis = tf.transpose(output, [1,0,2])
			last_timestamp 	= tf.gather(transpose_time_axis, int(transpose_time_axis.get_shape()[0]) - 1)
			init_weight 	= tf.Variable(tf.random_normal([self.hidden_units, self.num_output]))
			init_bias 		= tf.Variable(tf.random_normal([self.num_output]))
			self.__prediction = tf.nn.softmax(tf.matmul(last_timestamp, init_weight) + init_bias)
			self.__setObjective()

		def startTraining(self, data, labels, epoch, print_debug_steps=10):
			init_var = tf.global_variables_initializer()
			data_placeholder 	= self.__input_tensor
			labels_placeholder 	= self.__output_tensor	

			import h5py
			up_data = h5py.File('up.hdf5', 'r+')
			up_label = h5py.File('up_label.hdf5', 'r+')

			with tf.Session() as sess:
				sess.run(init_var)

				for _ in range(epoch):
					sess.run(self.__objective, feed_dict={data_placeholder: up_data['data'][0:], labels_placeholder: up_label['data'][0:]})
					loss = sess.run(self.__loss, feed_dict={data_placeholder: up_data['data'][0:], labels_placeholder: up_label['data'][0:]})
					print("Loss: {0}".format(loss))

if __name__ == "__main__":
	lstm = SingleLayerLSTMNetwork()
	lstm.setHiddenUnits(100).setInputSize(160).setOutputSize(12).setTimeSteps(49) \
		.setLossFunction('cross_enthropy').setOptimizer('gradient_descent').setLearningRate(0.05) \
		.build()
	lstm.startTraining(data=None, labels=None, epoch=1)