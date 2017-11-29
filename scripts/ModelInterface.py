from abc import abstractmethod

class ModelInterface(object):
	@abstractmethod
	def build(self):
		raise NotImplementedError("Class must implement the builder function build")

	@abstractmethod
	def loadModelParams(self):
		raise NotImplementedError("Class must implement the loading model parameters function")

	@abstractmethod
	def startTraining(self, data, labels):
		raise NotImplementedError("Class must implement the start training function")

	@abstractmethod
	def evaluate(self):
		raise NotImplementedError("Class must implement evaluation function")
