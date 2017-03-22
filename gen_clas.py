### General purpose classifier in Tensorflow
### Luke Taylor

# Image processing
from PIL import Image

import tensorflow as tf
import numpy as np
import random



### Go to line 158 to change input configs



class DnnClassifier:

	def __init__(self, inConfig):
		# Dataset filenames
		TrainingSet_File = inConfig.get('trainingset_file')
		TestSet_File 	 = inConfig.get('testset_file')

		# Load datasets.
		self.training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
			filename       = TrainingSet_File,
			target_dtype   = np.int,
			#features_dtype = np.float32)
			features_dtype = np.int)

		# print("******************")
		# print("** TRAINING SET **")
		# print(self.training_set)
		# print("******************")

		self.test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
			filename       = TestSet_File,
			target_dtype   = np.int,
			#features_dtype = np.float32)
			features_dtype = np.int)

		# Specify that all features have real-value data
		self._feature_columns = [tf.contrib.layers.real_valued_column( "",
			dimension = inConfig.get('dimension'))]


	def run(self, inConfig):
		# Build 3 layer DNN with 10, 20, 10 units respectively.
		self.classifier = tf.contrib.learn.DNNClassifier(
			feature_columns = self._feature_columns,
			hidden_units    = inConfig.get('hidden_units'),
			n_classes       = inConfig.get('n_classes'),
			model_dir       = inConfig.get('model_dir'))

		# Fit model
		self.classifier.fit(
			x = self.training_set.data,
			y = self.training_set.target,
			steps = inConfig.get('num_steps'))

		accuracy_score = self.classifier.evaluate(
			x = self.test_set.data,
			y = self.test_set.target)["accuracy"]
		
		#print('Accuracy: {0:f}'.format(accuracy_score))

		self.accuracy = accuracy_score	


	def predict(self, new_samples):
		y = list(self.classifier.predict(
				new_samples,
				as_iterable = True))
		
		# print('Predictions: {}'.format(str(y)))
		return y


def get_img_pixel_data():
	im = Image.open('one28by28.png', 'r')
	im_grey = im.convert('L') 

	bw = im_grey.point(lambda x: 255 if x < 128 else 0, '1') # black & white conversion

	width, height = bw.size
	pixel_values = list(bw.getdata())

	# print("Input length: " + str(len(pixel_values)))
	# print("W: " + str(width) + " H: " + str(height))
	# print("PV " + str(pixel_values))

	return pixel_values



def main():

	tf.reset_default_graph()

	### Configurations
	setup_config = {
		'trainingset_file'	: 'iris_training.csv',
		'testset_file'		: 'iris_test.csv',
		'dimension'			: 4,
		'hidden_units'		: [10, 20, 10], # 10,20,10 is "standard"
		'n_classes'			: 3,
		'model_dir'			: "/tmp/genclas_model",
		'num_steps'			: 2000
		}

	###	Transfusion Config
	tfuse_setup_config = {
		'trainingset_file'	: 'transfusion_train.csv',
		'testset_file'		: 'transfusion_test.csv',
		'dimension'			: 4,
		'hidden_units'		: [10, 20, 10], # 10,20,10 is "standard"
		'n_classes'			: 2,
		'model_dir'			: "/tmp/genclas_model_tfuse_2",
		'num_steps'			: 2000
		}

	###	MNIST Config
	mnist_config = {
		'trainingset_file'	: 'mnist_train.csv',
		'testset_file'		: 'mnist_test.csv',
		'dimension'			: 784, # 28px * 28px
		'hidden_units'		: [10, 20, 10], # 10,20,10 is "standard"
		'n_classes'			: 10,
		'model_dir'			: "/tmp/genclas_model_2",
		'num_steps'			: 20000
		}


	# Classify new samples.
	new_samples = np.array(
		[
			[6.4, 3.2, 4.5, 1.5],
			[5.8, 3.1, 5.0, 1.7]
		],
		dtype=float)

	tfuse_samples = np.array(
		[
			[5,  2,  500,   12],
			[14, 8,  2000,  46], 	## result = 0
			[11, 8,  2000,  52], 	## result = 1 - failing
			[7,  9,  2250,  89], 	## result = 1 - failing
			[2,  50, 12500, 98] 	## result = 1 - passing (from input set)
		],
		dtype=int)

	mnist_samples = np.array(
		[
			get_img_pixel_data()
		],
		dtype=int)


	###	Compute model
	###	CHANGE CONFIGS HERE VVV

	net = DnnClassifier(mnist_config)
	net.run(mnist_config)

	p = net.predict(mnist_samples)

	### END Change configs ^^^


	print ('Predictions: {}'.format(str(p)))
	print ("Accuracy: " + str(net.accuracy))


if __name__ == '__main__':
	main()