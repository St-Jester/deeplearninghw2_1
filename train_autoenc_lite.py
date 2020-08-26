from __future__ import print_function
import numpy as np
import pickle
from utils import get_loss, get_random_batch, images2batches, init_uniform, relu, diff_numpy, multiply_matrixes

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
        B_in = np.matmul(np.ones((BATCH_SIZE, 1)),
                         self.b_in.reshape(1, P))  # [20, 75]
        a_in = np.matmul(x, self.w_in.transpose()) + B_in  # [20, 75]
        z_in_numpy = relu(a_in)

        #
        # Please, add forward pass here
        #
        # layer rec
        B_rec = np.matmul(np.ones((BATCH_SIZE, 1)),
                          self.b_rec.reshape(1, P))
        a_rec = np.matmul(z_in_numpy, self.w_rec.transpose()) + B_rec  # [20, 75]
        z_rec_numpy = relu(a_rec)

        a_link = np.matmul(x[:, ::3], self.w_link.transpose())
        z_link = a_link

        # layer out
        x_out = z_link + z_rec_numpy

        B_out = np.matmul(np.ones((BATCH_SIZE, 1)),
                          self.b_out.reshape(1, D))
        a_out = np.matmul(x_out, self.w_out.transpose()) + B_out
        y = relu(a_out)
        return y

    def backprop(self, some_args):
        #
        # Please, add backpropagation pass here
        #
        return 0  # dw

    def apply_dw(self, dw):
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

#
# Main cycle
for i in range(UPDATES_NUM):
    # Get random batch for Stochastic Gradient Descent
    X_batch_train = get_random_batch(batches_train, BATCH_SIZE)

    # Forward pass, calculate network''s outputs
    Y_batch = neural_network.forward(X_batch_train)

    a = neural_network.forward_vector(X_batch_train)
    b = neural_network.forward_scalar(X_batch_train)
    print(diff_numpy(a, b))
    break
    # Calculate sum squared loss
    # loss = get_loss(Y_batch, X_batch_train)
    #
    # # Backward pass, calculate derivatives of loss w.r.t. weights
    # dw = neural_network.backprop(some_args)
    #
    # # Correct neural network''s weights
    # neural_network.apply_dw(dw)

#
# Load images_test.pickle here, run the network on it and show results here
#
# import matplotlib.pyplot as plt
# count, bins, ignored = plt.hist(neural_network.w_in[1], 15, density=True)
# plt.show()
