import tensorflow as tf 
# Chain of responsibility utilities script for defining various optimizers  

class OptimizerCORTail(object):
	# Must be at the end of the chain
	next_in_chain 	= None

	@classmethod
	def get(cls, identifier):
		return None

class GradientDescentOptimizer(object):
	next_in_chain 	= OptimizerCORTail()
	identifier 		= "gradient_descent"

	@classmethod
	def get(cls, identifier):
		if(identifier == cls.identifier):
			return tf.train.GradientDescentOptimizer
		return cls.next_in_chain.get(identifier)

class OptimizerCORHeader(object):
	next_in_chain 	= GradientDescentOptimizer()

	@classmethod
	def get(cls, identifier):
		return cls.next_in_chain.get(identifier)