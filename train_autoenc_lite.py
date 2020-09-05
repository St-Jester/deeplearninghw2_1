from __future__ import print_function
import numpy as np
import pickle
from utils import get_loss, get_random_batch, images2batches, init_uniform, relu, diff_numpy, multiply_matrixes, imshow
import time

BATCH_SIZE = 20
UPDATES_NUM = 1000
IMG_SIZE = 15
D = 225  # IMG_SIZE*IMG_SIZE
P = 75  # D /// 3
LEARNING_RATE = 0.001
np.random.seed(0)


class EncDecNetLite():
    def __init__(self):
        super(EncDecNetLite, self).__init__()

        self.w_in = np.zeros((P, D))
        self.w_out = np.zeros((D, P))
        self.w_link = np.zeros((P, P))

        self.w_rec = np.eye(P)

        self.b_in = np.zeros((1, P))
        self.b_rec = np.zeros((1, P))
        self.b_out = np.zeros((1, D))

        self.x = np.zeros((P, D))
        self.x_reduce = np.zeros((1, P))

        self.x_reduce = np.zeros((1, P))

        self.z_in = np.zeros((1, P))
        self.z_rec = np.zeros((1, P))
        self.z_link = np.zeros((1, P))

        self.a_in = np.zeros((1, P))
        self.a_rec = np.zeros((1, P))
        self.a_link = np.zeros((1, P))
        self.a_out = np.zeros((1, D))
        #
        # Please, add other weights here
        #

    def init(self):
        self.w_in = init_uniform(self.w_in)
        self.w_out = init_uniform(self.w_out)
        self.w_link = init_uniform(self.w_link)
        #
        # Please, add initializations of other weights here
        #

    def forward_vector(self, x):
        B_in = np.matmul(np.ones((BATCH_SIZE, 1)),
                         self.b_in.reshape(1, P))  # [20, 75]

        a_in = np.matmul(x, self.w_in.transpose()) + B_in  # [20, 75]

        z_in_numpy = relu(a_in)
        return z_in_numpy

    def forward_scalar(self, x):
        B_in_scalar = multiply_matrixes(np.ones((BATCH_SIZE, 1)), self.b_in)
        a_in_scalar = multiply_matrixes(x, self.w_in.transpose()) + B_in_scalar
        z_in_numpy = relu(a_in_scalar)

        return z_in_numpy

    def forward(self, x):
        # given for layer_in
        self.x = x
        B_in = np.matmul(np.ones((BATCH_SIZE, 1)),
                         self.b_in.reshape(1, P))  # [20, 75]
        self.a_in = np.matmul(self.x, self.w_in.transpose()) + B_in  # [20, 75]
        self.z_in = relu(self.a_in)

        #
        # Please, add forward pass here
        #
        # layer rec
        B_rec = np.matmul(np.ones((BATCH_SIZE, 1)),
                          self.b_rec.reshape(1, P))
        self.a_rec = np.matmul(self.z_in, self.w_rec.transpose()) + B_rec  # [20, 75]
        self.z_rec = relu(self.a_rec)

        self.x_reduce = x[:, ::3]
        self.a_link = np.matmul(self.x_reduce, self.w_link.transpose())
        # identity function
        self.z_link = self.a_link

        # layer out
        self.x_out = self.z_link + self.z_rec

        B_out = np.matmul(np.ones((BATCH_SIZE, 1)),
                          self.b_out.reshape(1, D))
        self.a_out = np.matmul(self.x_out, self.w_out.transpose()) + B_out

        y = relu(self.a_out)
        return y

    def backprop(self, Y_batch, X_batch_train):
        # calculate derivative for L_out (backward prop for L_out)

        # loss derivative
        der_out = 2 * (Y_batch - X_batch_train)
        # relu derivative: apply mask and "multiply by 0" those vals that are <0
        der_out[self.a_out < 0] = 0

        # dictionary for derivatives values
        dw_dic = dict()

        # weight updates for L_out der_out*z_out (where z_out is the same as x_out)
        delta_w_out = np.dot(der_out.T, self.x_out)
        dw_dic['delta_w_out'] = delta_w_out

        # biases updates = same as sum of der_out
        delta_b_out = np.sum(der_out, axis=0, keepdims=True)
        dw_dic['delta_b_out'] = delta_b_out

        # calculate derivative for L_link (backward prop for L_link)

        # has identity function so derivative is simple
        der_link = np.dot(der_out, self.w_out)
        delta_w_link = np.dot(der_link.T, self.x_reduce)
        dw_dic['delta_w_link'] = delta_w_link

        # derivative for L_rec (backward prop for L_rec) has ReLU. In Task it is not trainable
        der_rec = np.dot(der_out, self.w_out)
        der_rec[self.a_rec < 0] = 0

        # derivative for L_in (backward prop for L_in) has ReLU
        der_in = np.dot(der_rec, self.w_rec)
        der_in[self.a_in < 0] = 0

        delta_w_in = np.dot(der_in.T, self.x)
        dw_dic['delta_w_in'] = delta_w_in

        delta_b_in = np.sum(der_in, axis=0, keepdims=True)
        dw_dic['delta_b_in'] = delta_b_in

        #
        # Please, add backpropagation pass here
        #
        return dw_dic

    def apply_dw(self, dw):
        # W*E
        self.w_in -= LEARNING_RATE * dw['delta_w_in']
        self.b_in -= LEARNING_RATE * dw['delta_b_in']
        self.w_link -= LEARNING_RATE * dw['delta_w_link']
        self.w_out -= LEARNING_RATE * dw['delta_w_out']
        self.b_out -= LEARNING_RATE * dw['delta_b_out']
        #
        # Correct neural network''s weights
        #
        pass


# Load train data
images_train = pickle.load(open('images_train.pickle', 'rb'))
# # Convert images to batching-friendly format
batches_train = images2batches(images_train)

# Create neural network
neural_network = EncDecNetLite()
# Initialize weights
neural_network.init()

# make it True if need to compare. On test run on my machine comparison showed the following:
# Vector time 0.0840296745300293
# Scalar time 319.17129278182983
do_compare = False

# test speed
if do_compare:
    start_vector = time.time()
    for i in range(UPDATES_NUM):
        X_batch_train = get_random_batch(batches_train, BATCH_SIZE)
        a = neural_network.forward_vector(X_batch_train)
    end_vector = time.time()
    print(f"Vector time {end_vector - start_vector}")

    start_scalar = time.time()
    for i in range(UPDATES_NUM):
        X_batch_train = get_random_batch(batches_train, BATCH_SIZE)
        b = neural_network.forward_scalar(X_batch_train)
    end_scalar = time.time()
    print(f"Scalar time {end_scalar - start_scalar}")

# print(diff_numpy(a, b))

#
# Main cycle
loss_arr = []

for i in range(UPDATES_NUM):
    # Get random batch for Stochastic Gradient Descent
    X_batch_train = get_random_batch(batches_train, BATCH_SIZE)

    # Forward pass, calculate network''s outputs
    Y_batch = neural_network.forward(X_batch_train)

    # Calculate sum squared loss
    loss = get_loss(Y_batch, X_batch_train)
    loss_arr.append(loss)
    #
    # # Backward pass, calculate derivatives of loss w.r.t. weights
    dw = neural_network.backprop(Y_batch, X_batch_train)
    #
    # # Correct neural network''s weights
    neural_network.apply_dw(dw)

#
# Load images_test.pickle here, run the network on it and show results here
#
import matplotlib.pyplot as plt
plt.figure(1)
# Plot learning curve
plt.title("Learning curve plot")
plt.xlabel("Iteration")
plt.ylabel("Loss function")
plt.plot(np.arange(0, UPDATES_NUM), loss_arr)

print("\n")

# Load test
# images, run it through the network and report as pairs, inputs together
# with outputs
img_test = pickle.load(open('images_test.pickle', 'rb'))
print(img_test.shape)
batches_test = images2batches(img_test)
Y_test = neural_network.forward(batches_test)

print(Y_test.shape)
# make it (20,15,15)
Y_test = Y_test.reshape((img_test.shape[0], img_test.shape[1], img_test.shape[2]))


f2 = plt.figure(2)

plt.subplot(211)
plt.imshow(img_test[0], cmap='gray', vmin=0, vmax=255)
plt.title("before")
plt.subplot(212)
plt.title("after")

plt.imshow(Y_test[0], cmap='gray', vmin=0, vmax=255)
plt.show()
