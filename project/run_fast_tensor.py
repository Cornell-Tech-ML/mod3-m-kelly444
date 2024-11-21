"""
Implementation of the main training loop for fast tensor operations.
"""

import random
import numba
import minitorch
from minitorch import datasets
from typing import Callable, Optional, List

# Import dataset utilities and backend configurations
FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)

# Initialize GPUBackend as None
GPUBackend = None
if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def default_log_fn(epoch: int, total_loss: float, correct: int, losses: List[float]) -> None:
    """
    Default logging function for training progress.

    Args:
        epoch (int): Current training epoch
        total_loss (float): Total loss for the current epoch
        correct (int): Number of correct predictions
        losses (List[float]): Historical loss values
    """
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


def RParam(*shape: int, backend: minitorch.TensorBackend = FastTensorBackend) -> minitorch.Parameter:
    """
    Create a randomly initialized parameter tensor.

    Args:
        *shape: Variable length argument for tensor dimensions
        backend: Tensor backend to use (CPU or GPU)

    Returns:
        Parameter: Initialized parameter centered around zero (-0.5 to 0.5)
    """
    r = minitorch.rand(shape, backend=backend)
    return minitorch.Parameter(r - 0.5)


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
    def __init__(self, in_size: int, out_size: int, backend: minitorch.TensorBackend):
        super().__init__()
        self.weights = RParam(in_size, out_size, backend=backend)
        self.bias = RParam(out_size, backend=backend)
        self.out_size = out_size

    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        """
        Forward pass through the linear layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_size)

        Returns:
            Tensor: Output tensor of shape (batch_size, out_size)
        """
        batch_size = x.shape[0]
        return x @ self.weights.value + self.bias.value.view(1, self.out_size).expand((batch_size, self.out_size))


class Network(minitorch.Module):
    """
    Neural network architecture for binary classification.

    Implements a three-layer neural network with configurable hidden layer size.
    The network structure is: Input(2) -> Hidden -> Hidden -> Output(1)

    Args:
        hidden (int): Number of neurons in each hidden layer
        backend: Tensor backend to use for computations
    """
    def __init__(self, hidden: int, backend: minitorch.TensorBackend):
        super().__init__()
        self.layer1 = Linear(2, hidden, backend)     # Input layer
        self.layer2 = Linear(hidden, hidden, backend)  # Hidden layer
        self.layer3 = Linear(hidden, 1, backend)     # Output layer

    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 2)

        Returns:
            Tensor: Network predictions of shape (batch_size, 1)
        """
        # Apply ReLU activation after each layer except the last
        h1 = self.layer1.forward(x).relu()
        h2 = self.layer2.forward(h1).relu()
        return self.layer3.forward(h2).sigmoid()


class FastTrain:
    """
    Training manager for the neural network.

    Handles the training loop, optimization, and evaluation of the model.

    Args:
        hidden_layers (int): Number of neurons in hidden layers
        backend: Tensor backend to use (defaults to FastTensorBackend)
    """
    def __init__(self, hidden_layers: int, backend: minitorch.TensorBackend = FastTensorBackend):
        self.hidden_layers = hidden_layers
        self.backend = backend
        self.model = Network(hidden_layers, backend)

    def run_one(self, x: List[float]) -> minitorch.Tensor:
        """
        Run prediction for a single input sample.

        Args:
            x (List[float]): Single input sample

        Returns:
            Tensor: Model prediction
        """
        return self.model.forward(minitorch.tensor([x], backend=self.backend))

    def run_many(self, X: List[List[float]]) -> minitorch.Tensor:
        """
        Run predictions for multiple input samples.

        Args:
            X (List[List[float]]): Batch of input samples

        Returns:
            Tensor: Model predictions for the batch
        """
        return self.model.forward(minitorch.tensor(X, backend=self.backend))

    def train(
        self,
        data: datasets.Dataset,
        learning_rate: float,
        max_epochs: int = 500,
        log_fn: Optional[Callable[[int, float, int, List[float]], None]] = default_log_fn
    ) -> None:
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
        BATCH = 10
        losses = []

        for epoch in range(max_epochs):
            total_loss = 0.0
            # Shuffle training data
            data_list = list(zip(data.X, data.y))
            random.shuffle(data_list)
            X_shuf, y_shuf = zip(*data_list)

            # Mini-batch training
            for batch_start in range(0, len(X_shuf), BATCH):
                optim.zero_grad()

                # Prepare batch
                batch_end = batch_start + BATCH
                X = minitorch.tensor(X_shuf[batch_start:batch_end], backend=self.backend)
                y = minitorch.tensor(y_shuf[batch_start:batch_end], backend=self.backend)

                # Forward pass
                out = self.model.forward(X).view(y.shape[0])
                prob = (out * y) + (out - 1.0) * (y - 1.0)
                loss = -prob.log()
                loss = loss.sum()

                # Backward pass
                (loss / y.shape[0]).backward()

                total_loss = loss[0]

                # Update parameters
                optim.step()

            # Record loss
            losses.append(total_loss)

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs - 1:
                # Compute accuracy
                X = minitorch.tensor(data.X, backend=self.backend)
                y = minitorch.tensor(data.y, backend=self.backend)
                out = self.model.forward(X).view(y.shape[0])
                y_pred = minitorch.tensor([1.0 if o > 0.5 else 0.0 for o in out])
                correct = int((y_pred == y).sum()[0])

                if log_fn:
                    log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--PTS", type=int, default=50, help="number of points in dataset")
    parser.add_argument("--HIDDEN", type=int, default=10, help="number of hidden neurons")
    parser.add_argument("--RATE", type=float, default=0.05, help="learning rate")
    parser.add_argument("--BACKEND", default="cpu", help="backend to use (cpu/gpu)")
    parser.add_argument("--DATASET", default="simple", help="dataset to use")
    parser.add_argument("--PLOT", action="store_true", help="plot results")

    args = parser.parse_args()

    PTS = args.PTS

    # Load specified dataset
    if args.DATASET == "xor":
        data = datasets.Xor(PTS)
    elif args.DATASET == "simple":
        data = datasets.Simple(PTS)
    elif args.DATASET == "split":
        data = datasets.Split(PTS)
    elif args.DATASET == "diag":
        data = datasets.Diag(PTS)
    elif args.DATASET == "circle":
        data = datasets.Circle(PTS)
    elif args.DATASET == "spiral":
        data = datasets.Spiral(PTS)
    else:
        raise ValueError(f"Unknown dataset: {args.DATASET}")

    HIDDEN = args.HIDDEN
    RATE = args.RATE

    # Check if GPU backend is requested but not available
    if args.BACKEND == "gpu" and GPUBackend is None:
        print("GPU backend requested but CUDA is not available. Falling back to CPU backend.")
        backend = FastTensorBackend
    else:
        backend = GPUBackend if args.BACKEND == "gpu" else FastTensorBackend

    # Initialize and train the model
    fast_trainer = FastTrain(HIDDEN, backend=backend)
    fast_trainer.train(data, RATE)

    if args.PLOT:
        import matplotlib.pyplot as plt
        import numpy as np

        # Create visualization
        plt.figure(figsize=(10, 10))

        # Plot training data
        plt.scatter([x[0] for x in data.X], [x[1] for x in data.X],
                   c=['red' if y == 1 else 'blue' for y in data.y])

        # Create a grid of points for visualization
        x_range = np.linspace(min(x[0] for x in data.X) - 0.5,
                            max(x[0] for x in data.X) + 0.5, 100)
        y_range = np.linspace(min(x[1] for x in data.X) - 0.5,
                            max(x[1] for x in data.X) + 0.5, 100)
        xx, yy = np.meshgrid(x_range, y_range)
        grid_points = [[x, y] for x, y in zip(xx.ravel(), yy.ravel())]

        # Get predictions for grid points
        predictions = fast_trainer.run_many(grid_points)
        zz = predictions.detach().to_numpy().reshape(xx.shape)

        # Plot decision boundary
        plt.contour(xx, yy, zz, levels=[0.5], colors='black')
        plt.colorbar(plt.contourf(xx, yy, zz))

        plt.title(f'Dataset: {args.DATASET}, Hidden Units: {HIDDEN}')
        plt.show()