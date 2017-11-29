import tensorflow as tf 
# Chain of responsibility utilities script for defining various loss functions 

class LossCORTail(object):
	# Must be at the end of the chain
	next_in_chain 	= None

	@classmethod
	def get(cls, identifier):
		return None

class CrossEnthropyLoss(object):
	next_in_chain 	= LossCORTail()
	identifier 		= "cross_enthropy"

	@classmethod
	def get(cls, identifier):
		if(identifier == cls.identifier):
			return tf.nn.softmax_cross_entropy_with_logits
		return cls.next_in_chain.get(identifier)

class LossCORHeader(object):
	# Denotes the start of the chain
	next_in_chain 	= CrossEnthropyLoss()

	@classmethod
	def get(cls, identifier):
		return cls.next_in_chain.get(identifier)