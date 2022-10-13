import gzip
import struct
import sys

import numpy as np
import numpy_typing as npt

import beartype

sys.path.append("python/")
import needle as ndl


@beartype.beartype
def parse_mnist(
    image_filesname: str, label_filename: str
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    # Load images
    with gzip.open(image_filesname) as f:
        magic, n_images, rows, cols = struct.unpack(">IIII", f.read(16))
        print("magic number is", magic)
        # f already points to the first image
        image_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(
            n_images, rows * cols
        )
        images = image_data.astype(np.float32) / 255.0

    with gzip.open(label_filename) as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels


@beartype.beartype
def softmax_loss(Z: ndl.Tensor, y_one_hot: ndl.Tensor) -> ndl.Tensor:
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y_one_hot (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    # Take exp of the input
    exps = ndl.exp(Z)
    loss = ndl.log(ndl.summation(exps, axis=1)) - ndl.summation(Z * y_one_hot, axis=1)
    # TODO: check if this is the right way of getting the shape
    return ndl.summation(loss) / np.prod(loss.shape)


class Model:
    def __init__(self, W1: ndl.Tensor, W2: ndl.Tensor):
        self.W1 = W1
        self.W2 = W2

    def forward(self, X: ndl.Tensor):
        Z1 = ndl.relu(X @ self.W1)
        return Z1 @ self.W2


def _onehot(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    y_onehot = np.zeros_like(prediction)
    y_onehot[np.arange(y.shape[0]), y] = 1
    return y_onehot


def nn_epoch(
    X: np.ndarray,
    y: np.ndarray,
    W1: ndl.Tensor,
    W2: ndl.Tensor,
    lr: float = 0.1,
    batch: int = 100,
):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    model = Model(W1, W2)

    for i in range(X.shape[0] // batch):
        start = i * batch
        end = start + batch
        logits = model.forward(ndl.Tensor.make_const(X[start:end]))
        I_y = (ndl.Tensor.make_const, _onehot(y[start:end], logits))
        gradient = softmax_loss(logits, I_y)
        gradient.backward()


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
