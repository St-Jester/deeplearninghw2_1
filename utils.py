import numpy as np
import matplotlib.pyplot as plt


def diff_numpy(a, b, msg=None):
    """Shows differences between two tensors"""
    if a.shape != b.shape:
        print('Wrong shape!')
        print(a.shape)
        print(b.shape)
    else:
        diff = (np.sum(a - b)) ** 2
        if msg:
            print('%s diff = %1.6f' % (msg, diff.item()))
        else:
            print('diff = %1.6f' % diff.item())


def images2batches(images):
    """Converts images to convenient for batching form"""
    ndata, img_size, _ = images.shape
    return np.reshape(images, (ndata, img_size * img_size))


def imshow(img):
    """Show image using matplotlib"""
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


def init_uniform(a):
    """Makes iniform initialization of weight matrix (please, use
    numpy.random.uniform function or similar"""
    r = np.sqrt(6 / (a.shape[0] + a.shape[1]))
    s = np.random.uniform(-r, r, size=a.shape)
    return s


def relu(m):
    """Implements ReLU activation function"""
    return np.maximum(0, m)


def get_random_batch(batches_train, batch_size):
    """Outputs random batch of batch_size"""
    return batches_train[np.random.choice(batches_train.shape[0], batch_size, replace=False), :]


def get_loss(Y_batch, X_batch_train):
    """Claculates sum squared loss"""
    pass


def multiply_matrixes(A, B):

    result = np.zeros((A.shape[0], B.shape[1]))
    # iterating by row of A
    for i in range(len(A)):

        # iterating by coloum by B
        for j in range(len(B[0])):

            # iterating by rows of B
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    return result
