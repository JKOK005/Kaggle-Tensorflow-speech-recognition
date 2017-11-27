from abc import abstractmethod

class ModelInterface(object):
	@abstractmethod
	def build(self):
		raise NotImplementedError("Class must implement the builder function build")

	@abstractmethod
	def getModel(self):
		raise NotImplementedError("Class must implement the getModel method")