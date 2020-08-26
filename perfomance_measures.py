from __future__ import print_function
import numpy as np
import pickle
from utils import get_loss, get_random_batch, images2batches, init_uniform, relu
from train_autoenc_lite import *

# Create neural network
neural_network = EncDecNetLite()
# Initialize weights
neural_network.init()
from timeit import default_timer as timer

start = timer()
neural_network.forward_scalar()
end = timer()
print("Scalar time :", end - start)  # Time in seconds, e.g. 5.38091952400282

start = timer()
neural_network.forward_vector()
end = timer()
print("Scalar time :", end - start)  # Time in seconds, e.g. 5.38091952400282
