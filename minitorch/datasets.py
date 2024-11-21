from dataclasses import dataclass
from typing import List
import random
import math


@dataclass
class Dataset:
    """A class representing a dataset for binary classification.

    Attributes
    ----------
        X (List[List[float]]): List of input features
        y (List[int]): List of binary labels

    """

    X: List[List[float]]
    y: List[int]


def make_pts(N: int) -> List[List[float]]:
    """Generate N random points in the unit square."""
    return [[random.random(), random.random()] for _ in range(N)]


def simple(N: int = 50) -> Dataset:
    """Generate a simple dataset."""
    X = make_pts(N)
    y = [1 if x[0] < 0.5 else 0 for x in X]
    return Dataset(X, y)


def diag(N: int = 50) -> Dataset:
    """Generate a diagonal dataset."""
    X = make_pts(N)
    y = [1 if x[0] + x[1] < 0.5 else 0 for x in X]
    return Dataset(X, y)


def split(N: int = 50) -> Dataset:
    """Generate a split dataset."""
    X = make_pts(N)
    y = [1 if x[0] < 0.2 or x[0] > 0.8 else 0 for x in X]
    return Dataset(X, y)


def xor(N: int = 50) -> Dataset:
    """Generate an XOR dataset."""
    X = make_pts(N)
    y = [
        1 if ((x[0] < 0.5 and x[1] > 0.5) or (x[0] > 0.5 and x[1] < 0.5)) else 0
        for x in X
    ]
    return Dataset(X, y)


def circle(N: int = 50) -> Dataset:
    """Generate a circle dataset."""
    X = make_pts(N)
    y = [1 if (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 > 0.1 else 0 for x in X]
    return Dataset(X, y)


def spiral(N: int = 50) -> Dataset:
    """Generate a spiral dataset."""

    def spiral_x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def spiral_y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = []
    # First spiral
    for i in range(5, 5 + N // 2):
        t = 10.0 * (float(i) / (N // 2))
        X.append([spiral_x(t) + 0.5, spiral_y(t) + 0.5])

    # Second spiral
    for i in range(5, 5 + N // 2):
        t = 10.0 * (float(i) / (N // 2))
        X.append([spiral_y(-t) + 0.5, spiral_x(-t) + 0.5])

    labels = [0] * (N // 2) + [1] * (N // 2)
    return Dataset(X, labels)


# Dictionary mapping dataset names to their generator functions
datasets = {
    "simple": simple,
    "split": split,
    "xor": xor,
    "diag": diag,
    "circle": circle,
    "spiral": spiral,
}
