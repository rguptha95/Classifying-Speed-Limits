import tensorflow as tf
from IPython import display
from PIL import Image
import glob
import numpy as np
import cv2
import random

speed_limit_probs = []
prob_25 = np.array([1.0, 0.5, 0.25, 0.1, 0, 0, 0])
prob_30 = np.array([0.5, 1.0, 0.5, 0.25, 0.1, 0, 0])
prob_35 = np.array([.25, 0.5, 1., 0.5, .25, .1, 0])
prob_40 = np.array([.1, .25, .5, 1., .5, .25, .1])
prob_45 = np.array([0, .1, .25, .5, 1., .5, .25])
prob_50 = np.array([0, 0, .1, 0.25, .5, 1., .5])
prob_60 = np.array([0, 0, 0, .1, 0.25, .5, 1.])
outcome = [25, 30, 35, 40, 45, 50, 60]

data = np.genfromtxt("sign_data.csv", delimiter = ',')
speed_limits = data[:,1]

image_list = []
for i, filename in enumerate(glob.glob('25/*.png')):
	prob = np.zeros(7)
	prob = prob_25
	im_full = cv2.imread(filename, 0)
	resized_image = cv2.resize(im_full, (64, 60))
	# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
	# cv2.imshow('image', np.array(resized_image, dtype = np.uint8 ))
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	resized_image = resized_image.flatten()
	image_list.append(resized_image)
	speed_limit_probs.append(prob)

for i, filename in enumerate(glob.glob('35/*.png')):
	prob = np.zeros(7)
	prob = prob_35
	im_full = cv2.imread(filename, 0)
	resized_image = cv2.resize(im_full, (64, 60))
	# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
	# cv2.imshow('image', np.array(resized_image, dtype = np.uint8 ))
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	resized_image = resized_image.flatten()
	image_list.append(resized_image)
	speed_limit_probs.append(prob)

for i, filename in enumerate(glob.glob('40/*.png')):
	prob = np.zeros(7)
	prob = prob_40
	im_full = cv2.imread(filename, 0)
	resized_image = cv2.resize(im_full, (64, 60))
	# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
	# cv2.imshow('image', np.array(resized_image, dtype = np.uint8 ))
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	resized_image = resized_image.flatten()
	image_list.append(resized_image)
	speed_limit_probs.append(prob)

for i, filename in enumerate(glob.glob('45/*.png')):
	prob = np.zeros(7)
	prob = prob_45
	im_full = cv2.imread(filename, 0)
	resized_image = cv2.resize(im_full, (64, 60))
	# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
	# cv2.imshow('image', np.array(resized_image, dtype = np.uint8 ))
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	resized_image = resized_image.flatten()
	image_list.append(resized_image)
	speed_limit_probs.append(prob)

for i, filename in enumerate(glob.glob('50/*.png')):
	prob = np.zeros(7)
	prob = prob_50
	im_full = cv2.imread(filename, 0)
	resized_image = cv2.resize(im_full, (64, 60))
	# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
	# cv2.imshow('image', np.array(resized_image, dtype = np.uint8 ))
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	resized_image = resized_image.flatten()
	image_list.append(resized_image)
	speed_limit_probs.append(prob)

combined = list(zip(image_list, speed_limit_probs))
random.shuffle(combined)

image_list[:], speed_limit_probs[:] = zip(*combined)

training_indices = 900
testing_indices = 1100

images_training = image_list[:900] #Normalize between 0 and 1
speedLimit_training = speed_limit_probs[:900]

images_testing = image_list[900:1100] #Normalize between 0 and 1
speedLimit_testing = speed_limit_probs[900:1100]

#Hyper Parameters
learning_rate = 1e-4
num_iterations = 3000

#How many bins is our data encoded into:
# bins = speedLimit_training.shape[1]

#Setup tf placeholders for images and angles
im = tf.placeholder(tf.float32, (None, 60*64))
spl = tf.placeholder(tf.float32, (None, 7))

#Reshape input images into vectors
im_reshaped = tf.reshape(im, (-1, 64*60))

#Hidden Layer
a1 = tf.layers.dense(im_reshaped, 4, activation = tf.nn.relu)

#Output layer
yhat = tf.layers.dense(a1, 7)

#Loss function
loss = tf.losses.mean_squared_error(spl, yhat)

#Optimizer
train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

#Store errors as we train in a list:
RMSEs = []

#Initialize Tensorflow session:
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#And train!
for i in range(num_iterations):
    #Train Step
    sess.run(train_op, feed_dict = {im:(images_training), spl:speedLimit_training})
    #Measure error
    RMSEs.append(np.sqrt(loss.eval(feed_dict = {im:images_training, spl:speedLimit_training})))
    print("Iteration: " + str(i))

# print(RMSEs)

#Network Predictions on Testing Images:
yhat_array = yhat.eval(feed_dict = {im:images_testing})

#Center values of our prediction bins:
centers = np.linspace(0, 7, 7)

# Convert probabilistic prediction to a single angle
predicted_angles = []
for i in range(yhat_array.shape[0]):
    #Just pick bin with largest value:
    print("Prediction: " + str(outcome[np.argmax(yhat_array[i, :])]))
    print("Actual: " + str(outcome[np.argmax(speedLimit_testing[i])]))

sess.close()