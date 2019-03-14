import os
import numpy as np
import pickle
import random
import tensorflow as tf
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''
(1) process images (input, batch, format conversion, etc.)
(2) perform training (with proper batching)
(3) estimate performance (i.e., error rate and run time of each epoch)
(4) output (save) the final network parameters
'''

# dataset: CIFAR10
# 10 classes: (airplane, auto, bird, cat, deer, dog, frog, horse, ship, truck)
TRAIN_DATASETS = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5' ]
FOLDER_NAME = 'cifar-10-batches-py/'
SAVE_PATH = '/tmp/model.ckpt'
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_DEPTH = 3

# CNN:
'''
layer 1: CNN(relu, max pooling)
layer 2: CNN(relu, max pooling)
layer 31: CNN(relu)
layer 32: CNN(relu, max pooling)
layer 4: FNN
layer 5: output layer
'''
LABELS = 10
BATCH_SIZE = 100
EPOCH = 50
FILTER_SIZE_1 = 7
FILTER_SIZE_2 = 3
FILTER_SIZE_31 = 3
FILTER_SIZE_32 = 3
FILTER_DEPTH_1 = BATCH_SIZE
FILTER_DEPTH_2 = FILTER_DEPTH_1*2
FILTER_DEPTH_31 = FILTER_DEPTH_2*2
FILTER_DEPTH_32 = FILTER_DEPTH_31
HIDDEN_CELLS = 64*BATCH_SIZE
LR = 0.0001
# STEPS = 10001
# learning_rates = [0.001, 0.0015, 0.002, 0.0025]


def get_label_name(label):
	"""
	Load the label names from file
	"""
	index = np.argmax(label)
	labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	return labels[int(index)]

def one_hot_encode(np_array, num_label):
	temp = (np.arange(num_label) == np_array[:,None]).astype(np.float32)
	return temp

def reformat_data(dataset, label):
	np_dataset_ = dataset.reshape((len(dataset), IMAGE_DEPTH, IMAGE_WIDTH, IMAGE_HEIGHT)).transpose(0, 2, 3, 1)
	num_label = len(np.unique(label))
	np_label_ = one_hot_encode(np.array(label, dtype=np.float32), num_label)
	np_dataset, np_label = randomize(np_dataset_, np_label_)
	return np_dataset, np_label

def randomize(dataset, label):
	permutation = np.random.permutation(label.shape[0])
	shuffled_dataset = dataset[permutation, :, :]
	shuffled_label = label[permutation]
	return shuffled_dataset, shuffled_label

def data_process():
	current_path = os.path.dirname(os.path.abspath(__file__))
	file_path = current_path + '/' + FOLDER_NAME
	# load train and test dictionary
	train_dataset, train_label = [], []
	for dataset in TRAIN_DATASETS:
		with open(file_path + dataset, 'rb') as f0:
			train_dict = pickle.load(f0, encoding = 'bytes')
			train_dataset_temp, train_label_temp = train_dict[b'data'], train_dict[b'labels']
			train_dataset.append(train_dataset_temp)
			train_label += train_label_temp

	train_dataset = np.concatenate(train_dataset, axis = 0)
	train_dataset, train_label = reformat_data(train_dataset, train_label)
	print("training dataset contains {} images, each of size {}".format(train_dataset[:,:,:,:].shape[0], train_dataset[:,:,:,:].shape[1:]))
	return train_dataset, train_label


def flatten_tf_array(array):
    shape = array.get_shape().as_list()
    return tf.reshape(array, [-1, shape[1] * shape[2] * shape[3]])

def accuracy(pred, label):
	return (100.0 * np.sum(np.argmax(pred, 1) == np.argmax(label, 1)) / pred.shape[0])

def init_variables(filter_size1 = FILTER_SIZE_1, filter_depth1 = FILTER_DEPTH_1, 
				   filter_size2 = FILTER_SIZE_2, filter_depth2 = FILTER_DEPTH_2,
				   filter_size31 = FILTER_SIZE_31, filter_depth31 = FILTER_DEPTH_31, 
				   filter_size32 = FILTER_SIZE_32, filter_depth32 = FILTER_DEPTH_32,
				   num_hidden = HIDDEN_CELLS,
				   image_width = IMAGE_WIDTH, image_height = IMAGE_HEIGHT, image_depth = IMAGE_DEPTH, num_labels = LABELS):

	initializer = tf.contrib.layers.xavier_initializer()

	w1 = tf.Variable(initializer([filter_size1, filter_size1, image_depth, filter_depth1]), name='w1')
	b1 = tf.Variable(initializer([filter_depth1]))

	w2 = tf.Variable(initializer([filter_size2, filter_size2, filter_depth1, filter_depth2]))
	b2 = tf.Variable(initializer([filter_depth2]))

	w31 = tf.Variable(initializer([filter_size31, filter_size31, filter_depth2, filter_depth31]))
	b31 = tf.Variable(initializer([filter_depth31]))

	w32 = tf.Variable(initializer([filter_size32, filter_size32, filter_depth31, filter_depth32]))
	b32 = tf.Variable(initializer([filter_depth32]))

	w4 = tf.Variable(initializer([num_hidden, num_labels]))
	b4 = tf.Variable(initializer([num_labels]))

	variables = {
		'w1': w1, 'w2': w2, 'w31': w31, 'w32': w32, 'w4': w4, 
		'b1': b1, 'b2': b2, 'b31': b31, 'b32': b32, 'b4': b4
	}
	return variables

def model_cnn(data, variables):

	layer1_conv = tf.nn.conv2d(data, variables['w1'], [1, 1, 1, 1], padding='SAME')
	layer1_actv = tf.nn.relu(layer1_conv + variables['b1'])
	layer1_pool = tf.nn.max_pool(layer1_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

	layer2_conv = tf.nn.conv2d(layer1_pool, variables['w2'], [1, 1, 1, 1], padding='SAME')
	layer2_actv = tf.nn.relu(layer2_conv + variables['b2'])
	layer2_pool = tf.nn.max_pool(layer2_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

	layer31_conv = tf.nn.conv2d(layer2_pool, variables['w31'], [1, 1, 1, 1], padding='SAME')
	layer31_actv = tf.nn.relu(layer31_conv + variables['b31'])

	layer32_conv = tf.nn.conv2d(layer31_actv, variables['w32'], [1, 1, 1, 1], padding='SAME')
	layer32_actv = tf.nn.relu(layer32_conv + variables['b32'])
	layer32_pool = tf.nn.max_pool(layer32_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

	flat_layer = flatten_tf_array(layer32_pool)
	layer4_fccd = tf.matmul(flat_layer, variables['w4']) + variables['b4']
	
	return layer4_fccd

train_dataset, train_label= data_process()
print('learning_rate:', LR)
graph = tf.Graph()

with graph.as_default():
	tf_train_dataset = tf.placeholder(tf.float32, shape = (None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH), name='input_x')
	tf_train_label = tf.placeholder(tf.float32, shape = (BATCH_SIZE, LABELS))

	# initilization of weight and bias
	variables_ = init_variables
	variables = variables_(image_width = IMAGE_WIDTH, image_height = IMAGE_HEIGHT, image_depth = IMAGE_DEPTH, num_labels = LABELS)

	# initialize model
	model = model_cnn
	logits = model(tf_train_dataset, variables)

	# loss
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf_train_label))

	# optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate = LR).minimize(loss)

	# prediction for training set
	train_pred = tf.nn.softmax(logits, name='pred')

with tf.Session(graph = graph) as session:

	tf.global_variables_initializer().run()
	saver = tf.train.Saver()

	total_step = train_label.shape[0] // BATCH_SIZE
	for epoch in range(EPOCH):
		print("------------------------------- EPOCH {:02d} -------------------------------".format(epoch))
		# shuffle the dataset for training each epoch
		# train_dataset, train_label = randomize(train_dataset, train_label)
		epoch_time = 0

		for step in range(total_step):
			offset = (step * BATCH_SIZE) % (train_label.shape[0] - BATCH_SIZE)
			data = train_dataset[offset:(offset + BATCH_SIZE), :, :, :]
			label = train_label[offset:(offset + BATCH_SIZE), :]
			feed_dict = {tf_train_dataset: data, tf_train_label:label}

			# run time
			start_time = time.time()
			_, cost, pred = session.run([optimizer, loss, train_pred], feed_dict = feed_dict)
			runtime = time.time() - start_time
			epoch_time += runtime

			# error rate
			train_accuracy = accuracy(pred, label)
			if step % 200 == 0:
				summary = "batch {:04d}: loss is {:06.2f}, error rate on training set {:02.2f} %".format(step, cost, 100 - train_accuracy)
				print(summary)

		epoch_summary = "EPOCH {:02d}: run time {:.2f}min, accuracy on training set {:02.2f}%,  error rate {:02.2f}%".format(epoch, epoch_time / 60.0, train_accuracy,100 - train_accuracy)
		print('\n')
		print(epoch_summary)

	save_model = saver.save(session, CURRENT_PATH + SAVE_PATH)
	print("Model saved in path: %s" % save_model)





