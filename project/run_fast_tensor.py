import random
import numba
import minitorch
from minitorch import datasets

# Import dataset utilities and backend configurations
datasets = minitorch.datasets
FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)

# Initialize GPUBackend as None
GPUBackend = None
if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def default_log_fn(epoch, total_loss, correct, losses):
    """
    Default logging function for training progress.

    Args:
        epoch (int): Current training epoch
        total_loss (float): Total loss for the current epoch
        correct (int): Number of correct predictions
        losses (list): Historical loss values
    """
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


def RParam(*shape, backend):
    """
    Create a randomly initialized parameter tensor.

    Args:
        *shape: Variable length argument for tensor dimensions
        backend: Tensor backend to use (CPU or GPU)

    Returns:
        Parameter: Initialized parameter centered around zero (-0.5 to 0.5)
    """
    r = minitorch.rand(shape, backend=backend) - 0.5
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    """
    Neural network architecture for binary classification.

    Implements a three-layer neural network with configurable hidden layer size.
    The network structure is: Input(2) -> Hidden -> Hidden -> Output(1)

    Args:
        hidden (int): Number of neurons in each hidden layer
        backend: Tensor backend to use for computations
    """
    def __init__(self, hidden, backend):
        super().__init__()

        # Initialize three linear layers
        self.layer1 = Linear(2, hidden, backend)  # Input layer
        self.layer2 = Linear(hidden, hidden, backend)  # Hidden layer
        self.layer3 = Linear(hidden, 1, backend)  # Output layer

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 2)

        Returns:
            Tensor: Network predictions of shape (batch_size, 1)
        """
        # Apply ReLU activation after each layer except the last
        hidden1 = self.layer1.forward(x).relu()
        hidden2 = self.layer2.forward(hidden1).relu()
        output = self.layer3.forward(hidden2).sigmoid()
        return output


class Linear(minitorch.Module):
    """
    Linear (fully connected) neural network layer.

    Implements y = Wx + b transformation where W is the weight matrix
    and b is the bias vector.

    Args:
        in_size (int): Number of input features
        out_size (int): Number of output features
        backend: Tensor backend to use for computations
    """
    def __init__(self, in_size, out_size, backend):
        super().__init__()

        # Initialize weights with random values
        self.weights = RParam(in_size, out_size, backend=backend)

        # Initialize bias with small positive values
        s = minitorch.zeros((out_size,), backend=backend)
        s = s + 0.1
        self.bias = minitorch.Parameter(s)
        self.out_size = out_size

    def forward(self, x):
        """
        Forward pass through the linear layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_size)

        Returns:
            Tensor: Output tensor of shape (batch_size, out_size)
        """
        # Compute matrix multiplication between input and weights
        out = x @ self.weights.value
        # Add bias term to each output neuron
        return out + self.bias.value


class FastTrain:
    """
    Training manager for the neural network.

    Handles the training loop, optimization, and evaluation of the model.

    Args:
        hidden_layers (int): Number of neurons in hidden layers
        backend: Tensor backend to use (defaults to FastTensorBackend)
    """
    def __init__(self, hidden_layers, backend=FastTensorBackend):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers, backend)
        self.backend = backend

    def run_one(self, x):
        """
        Run prediction for a single input sample.

        Args:
            x: Single input sample

        Returns:
            Tensor: Model prediction
        """
        return self.model.forward(minitorch.tensor([x], backend=self.backend))

    def run_many(self, X):
        """
        Run predictions for multiple input samples.

        Args:
            X: Batch of input samples

        Returns:
            Tensor: Model predictions for the batch
        """
        return self.model.forward(minitorch.tensor(X, backend=self.backend))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        """
        Train the neural network.

        Implements mini-batch stochastic gradient descent with binary cross-entropy loss.

        Args:
            data: Training dataset containing X (features) and y (labels)
            learning_rate (float): Learning rate for optimization
            max_epochs (int): Maximum number of training epochs
            log_fn (callable): Function for logging training progress
        """
        # Initialize new model and optimizer
        self.model = Network(self.hidden_layers, self.backend)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        BATCH = 10  # Mini-batch size
        losses = []  # Track loss history

        for epoch in range(max_epochs):
            total_loss = 0.0

            # Shuffle data for each epoch
            c = list(zip(data.X, data.y))
            random.shuffle(c)
            X_shuf, y_shuf = zip(*c)

            # Mini-batch training
            for i in range(0, len(X_shuf), BATCH):
                optim.zero_grad()

                # Prepare batch
                X = minitorch.tensor(X_shuf[i : i + BATCH], backend=self.backend)
                y = minitorch.tensor(y_shuf[i : i + BATCH], backend=self.backend)

                # Forward pass
                out = self.model.forward(X).view(y.shape[0])

                # Calculate binary cross-entropy loss
                prob = (out * y) + (out - 1.0) * (y - 1.0)
                loss = -prob.log()

                # Backward pass
                (loss / y.shape[0]).sum().view(1).backward()

                total_loss = loss.sum().view(1)[0]

                # Update parameters
                optim.step()

            losses.append(total_loss)

            # Log progress every 10 epochs
            if epoch % 10 == 0 or epoch == max_epochs:
                # Evaluate on full dataset
                X = minitorch.tensor(data.X, backend=self.backend)
                y = minitorch.tensor(data.y, backend=self.backend)
                out = self.model.forward(X).view(y.shape[0])
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--PTS", type=int, default=50, help="number of points in dataset")
    parser.add_argument("--HIDDEN", type=int, default=10, help="number of hidden neurons")
    parser.add_argument("--RATE", type=float, default=0.05, help="learning rate for optimization")
    parser.add_argument("--BACKEND", default="cpu", help="backend mode (cpu or gpu)")
    parser.add_argument("--DATASET", default="simple", help="dataset type (simple, xor, or split)")
    parser.add_argument("--PLOT", default=False, help="enable plotting")

    args = parser.parse_args()
    PTS = args.PTS

    # Load specified dataset

    if args.DATASET == "xor":
        data = datasets["Xor"](PTS)
    elif args.DATASET == "simple":
        data = datasets["Simple"](PTS)
    elif args.DATASET == "split":
        data = datasets["Split"](PTS)
    elif args.DATASET == "diag":
        data = datasets["Diag"](PTS)
    elif args.DATASET == "circle":
        data = datasets["Circle"](PTS)
    elif args.DATASET == "spiral":
        data = datasets["Spiral"](PTS)

    HIDDEN = int(args.HIDDEN)
    RATE = args.RATE

    # Check if GPU backend is requested but not available
    if args.BACKEND == "gpu" and GPUBackend is None:
        print("GPU backend requested but CUDA is not available. Falling back to CPU backend.")
        backend = FastTensorBackend
    else:
        backend = GPUBackend if args.BACKEND == "gpu" else FastTensorBackend

    # Initialize and train the model
    FastTrain(HIDDEN, backend=backend).train(data, RATE)